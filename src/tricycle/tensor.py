import logging
import numbers
import uuid
import weakref
from typing import Callable, List, Optional, Sequence, Union

import numpy as np
from numpy.typing import ArrayLike

from tricycle import CUPY_ENABLED
from tricycle.exceptions import GPUDisabledException

logger = logging.getLogger(__name__)

Op = Callable[..., "Tensor"]


class Tensor:
    """
    An N-dimensional grid of numbers. This is implemented as a subclass
    of a standard numpy array
    """

    _id: int
    array: np.ndarray | ArrayLike
    args: tuple["Tensor", ...] | None = None
    back_fns: tuple[Op, ...] | None = None
    parents: set["Tensor"] | None = None
    grad: Optional["Tensor"] = None
    name: Optional[str] = None
    requires_grad: bool = False
    is_vector: bool = False

    def __init__(
        self,
        data: np.ndarray | ArrayLike,
        requires_grad: bool = False,
        is_vector: bool = False,
        name: str | None = None,
        _id: int | None = None,
    ):
        self._id = _id or uuid.uuid4().int
        if CUPY_ENABLED:
            import cupy

            if isinstance(data, (np.ndarray, cupy.ndarray)):
                self.array = data
            else:
                self.array = np.array(data)
        else:
            self.array = np.array(data)

        self.requires_grad = requires_grad
        self.is_vector = is_vector
        self.name = name

    def _attach_parents(self):
        """
        Traverse through the graph, labelling each tensor with the tensors that
        are direct parents to it in the graph
        """
        stack: list["Tensor"] = [self]

        while stack:
            node = stack.pop()

            if not node.args:
                continue

            for arg in node.args:
                if not arg.requires_grad:
                    continue

                if arg.parents is None:
                    # if we use a set, we get a circular reference
                    # which can't be garbage collected, leading to a memory
                    # leak so we need to do a weakref to avoid the circular
                    # reference
                    arg.parents = weakref.WeakSet()

                if node not in arg.parents:
                    stack.append(arg)
                    arg.parents.add(node)

    def _calculate_gradients(self, clip: float | None = None):
        """
        Traverse through the graph, calculating gradients along the way such
        that a child is only visited if the entirety of its parent's gradient
        has been computed
        """
        self.grad = to_tensor(
            self.xp.ones(self.array.shape, dtype=self.dtype),
            requires_grad=False,
            is_vector=self.is_vector,
        )

        stack: list["Tensor"] = [self]

        while stack:
            node = stack.pop()

            if node.args is None or node.back_fns is None:
                continue

            for arg, back_fns in zip(node.args, node.back_fns):
                if not arg.requires_grad:
                    continue

                if arg.parents is None:
                    raise ValueError(
                        "arg.parents is None. Parents must be attached",
                        "before calculating gradients. Did you forget to ",
                        "call _attach_parents?",
                    )

                # already visited along this edge, dont do it again
                if node not in arg.parents:
                    continue

                arg.parents.remove(node)

                # calculate gradients
                try:
                    grad = back_fns(node.grad)

                    # gradient clipping
                    if clip is not None:
                        grad.array = grad.xp.clip(grad.array, -clip, clip)

                    # add gradient
                    if arg.grad is None:
                        arg.grad = grad
                    else:
                        arg.grad.array += grad.array

                except Exception as e:
                    raise e

                # only move to arg if we have been to all of its parents
                if len(arg.parents) == 0:
                    # get rid of the weakref so we can pickle the model
                    arg.parents = None
                    stack.append(arg)

    def cleanup(self):
        """
        Traverse through the graph, deleting all non-parameter nodes in
        the graph to avoid a memory leak
        """
        stack: list["Tensor"] = [self]
        while stack:
            node = stack.pop()

            # add children to stack
            if node.args:
                stack.extend(iter(node.args))
                del node.args
            else:
                continue

            # delete node
            if hasattr(node, "grad") and node.grad is not None:
                del node.grad
            del node

    def backward(self, clip: float | None = None):
        """
        Perform a backward pass through the graph, calculating the gradient
        for each parameter
        """
        self._attach_parents()
        self._calculate_gradients(clip=clip)

    def __hash__(self) -> int:
        return self._id

    def __add__(self, other: Union[float, "Tensor"]) -> "Tensor":
        if isinstance(other, numbers.Number):
            from tricycle.unary import UnaryAdd

            return UnaryAdd()(self, other)
        elif isinstance(other, Tensor):
            from tricycle.binary import BinaryAdd

            return BinaryAdd()(self, other)
        else:
            raise NotImplementedError(
                f"Cannot add {type(self)} and {type(other)}"
            )

    def __radd__(self, other):
        return self + other

    def __iadd__(self, other):
        return self + other

    def __sub__(self, other):
        if isinstance(other, self.xp.ndarray) and not isinstance(
            other, Tensor
        ):
            other = to_tensor(other)
        if self.xp.isscalar(other):
            from tricycle.unary import UnarySubtract

            return UnarySubtract()(self, other)
        elif isinstance(other, Tensor):
            from tricycle.binary import BinarySubtract

            return BinarySubtract()(self, other)

        else:
            raise NotImplementedError(
                f"Cannot sub {type(self)} and {type(other)}"
            )

    def __rsub__(self, other):
        return -(self - other)

    def __isub__(self, other):
        return self.__sub__(other)

    def __mul__(self, other):
        if isinstance(other, self.xp.ndarray) and not isinstance(
            other, Tensor
        ):
            other = to_tensor(other)
        if self.xp.isscalar(other) or other.shape == ():
            from tricycle.unary import UnaryMultiply

            return UnaryMultiply()(self, other)

        elif isinstance(other, Tensor):
            from tricycle.binary import BinaryMultiply

            return BinaryMultiply()(self, other)

        else:
            raise NotImplementedError(
                f"Cannot mul {type(self)} and {type(other)}"
            )

    def __rmul__(self, other):
        return self * other

    def __imul__(self, other):
        return self * other

    def __neg__(self):
        return self * -1

    def __truediv__(self, other):
        if self.xp.isscalar(other):
            from tricycle.unary import UnaryMultiply

            return UnaryMultiply()(self, 1 / other)
        elif isinstance(other, Tensor):
            from tricycle.binary import BinaryDivide

            return BinaryDivide()(self, other)

        else:
            raise NotImplementedError(
                f"Cannot divide {type(self)} and {type(other)}"
            )

    def __rtruediv__(self, other):
        if self.xp.isscalar(other):
            from tricycle.unary import UnaryDivide

            return UnaryDivide()(other, self)
        elif isinstance(other, Tensor):
            from tricycle.binary import BinaryDivide

            return BinaryDivide()(other, self)

    def __itruediv__(self, other):
        return self / other

    def __floordiv__(self, _):
        raise NotImplementedError("Cannot floor divide")

    def __rfloordiv__(self, _):
        raise NotImplementedError("Cannot floor divide")

    def __ifloordiv__(self, _):
        raise NotImplementedError("Cannot floor divide")

    def __mod__(self, _):
        raise NotImplementedError("Cannot mod")

    def __pow__(self, other) -> "Tensor":
        if isinstance(other, self.xp.ndarray) and not isinstance(
            other, Tensor
        ):
            other = to_tensor(other)
        if self.xp.isscalar(other):
            from tricycle.unary import UnaryPower

            return UnaryPower()(self, other)
        elif isinstance(other, Tensor):
            raise NotImplementedError(
                "Cannot power two tensors of shape: "
                f"{self.shape}, {other.shape}"
            )
        else:
            raise NotImplementedError(
                f"Cannot power {type(self)} and {type(other)}"
            )

    def __lt__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.array < other.array)
        return Tensor(self.array < other)

    def __le__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.array <= other.array)
        return Tensor(self.array <= other)

    def __eq__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.array == other.array)
        return Tensor(self.array == other)

    def __ne__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.array != other.array)
        return Tensor(self.array != other)

    def __gt__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.array > other.array)
        return Tensor(self.array > other)

    def __ge__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.array >= other.array)
        return Tensor(self.array >= other)

    def __repr__(self):
        name = f", name={self.name}" if self.name is not None else ""
        return f"Tensor({self.array.__str__()}{name})"

    def __getitem__(self, idx):
        return to_tensor(self.array[idx], requires_grad=self.requires_grad)

    def __setitem__(self, idx, value):
        self.array[idx] = value

    @property
    def xp(self):
        return select_backend(self.array)

    def e(self, subscript: str) -> "Tensor":
        """
        Perform an einsum operation on the tensor
        """
        from tricycle.einsum import Einsum

        return Einsum(subscript)(self)

    def repeat(self, n_repeats: int) -> "Tensor":
        from tricycle.ops import Repeat

        return Repeat()(self, n_repeats)

    @property
    def shape(self) -> Sequence[int]:
        return self.array.shape

    @property
    def ndim(self) -> int:
        return self.array.ndim

    @property
    def dtype(self) -> np.dtype:
        return self.array.dtype

    def reshape(self, shape: Sequence[int]) -> "Tensor":
        from tricycle.ops import Reshape

        return Reshape()(self, shape)

    def split(self, n_splits: int, axis: int = -1) -> List["Tensor"]:
        from tricycle.ops import Split

        return Split()(self, n_splits=n_splits, axis=axis)

    def mean(self) -> "Tensor":
        divisor = self.shape[-1] if self.shape else 1
        return self.sum() / divisor

    def sum(self) -> "Tensor":
        from tricycle.unary import UnarySum

        # if self.is_vector:
        #     indices = "abcdefghijklmnopqrstuvwxy"[: self.ndim - 1]
        # else:
        #     indices = "abcdefghijklmnopqrstuvwxy"[: self.ndim]
        # return self.e(f"{indices}->")
        return UnarySum()(self)

    def close_to(
        self,
        other: Union["Tensor", ArrayLike, float, int],
        equal_nan=False,
        rtol=1e-4,
        **kwargs,
    ) -> bool:
        """
        Convenience method to check if two tensors are identical
        to within some tolerance
        """
        if not isinstance(other, Tensor):
            return self.xp.allclose(
                self.array,
                self.xp.array(other),
                equal_nan=equal_nan,
                rtol=rtol,
                **kwargs,
            )
        return self.xp.allclose(
            self.array, other.array, equal_nan=equal_nan, rtol=rtol, **kwargs
        )

    def to_vector(self):
        """
        Treat this tensor as a vector
        """
        return vectorise(self)

    def from_vector(self):
        """
        Treat a vectorised tensor as a normal tensor
        """
        return unvectorise(self)

    @property
    def on_gpu(self):
        if not CUPY_ENABLED:
            return False
        import cupy

        return isinstance(self.array, cupy.ndarray)

    def to_gpu(self, device: int = 0):
        """
        Move this tensor to the GPU
        """
        if not CUPY_ENABLED:
            raise GPUDisabledException(
                "Cannot move tensor to GPU because CuPY is not enabled"
            )
        import cupy

        cupy.cuda.Device(device).use()
        self.array = cupy.asarray(self.array)
        return self

    def from_gpu(self):
        """
        Move this tensor from the GPU
        """
        if not CUPY_ENABLED:
            raise GPUDisabledException(
                "Cannot move tensor from GPU because CuPY is not enabled"
            )
        import cupy

        self.array = cupy.asnumpy(self.array)
        return self

    def zero_grad(self):
        self.grad = None
        self.args = None
        self.back_fns = None

        return self

    def numpy(self):
        if not CUPY_ENABLED:
            return self.array

        import cupy

        return cupy.asnumpy(self.array) if self.on_gpu else self.array


def to_tensor(
    tensor_like: ArrayLike,
    name: Optional[str] = None,
    requires_grad: bool = True,
    is_vector: bool = False,
    _id: int | None = None,
    dtype: np.dtype | None = np.float32,
    **kwargs,
) -> Tensor:
    """
    Create a new Tensor instance. First, we convert the argument to a numpy
    array and then to a tensor
    """
    if CUPY_ENABLED:
        import cupy

        if isinstance(tensor_like, Tensor):
            array = tensor_like.array
        elif isinstance(tensor_like, (np.ndarray, cupy.ndarray)):
            array = tensor_like
            if dtype is not None:
                array = array.astype(dtype)
        else:
            if dtype is None:
                dtype = np.float32
            array = np.asarray(tensor_like, dtype=dtype, **kwargs)

    elif isinstance(tensor_like, Tensor):
        array = tensor_like.array
    else:
        array = np.asarray(tensor_like, dtype=dtype, **kwargs)

    return Tensor(
        array,
        name=name,
        requires_grad=requires_grad,
        is_vector=is_vector,
        _id=_id,
    )


def vectorise(tensor: Tensor) -> Tensor:
    """
    Tell Tricycle to treat this tensor as a group of vectors
    """
    if tensor.is_vector:
        return tensor

    result = to_tensor(
        tensor.array,
        is_vector=True,
        requires_grad=tensor.requires_grad,
        dtype=tensor.array.dtype,
    )
    result.args = (tensor,)
    result.back_fns = (unvectorise,)
    return result


def unvectorise(tensor: Tensor) -> Tensor:
    """
    Tell Tricycle to treat this tensor as a single tensor
    (not a group of vectors)
    """
    if not tensor.is_vector:
        return tensor

    result = to_tensor(
        tensor.array,
        is_vector=False,
        requires_grad=tensor.requires_grad,
        dtype=tensor.array.dtype,
    )
    result.args = (tensor,)
    result.back_fns = (vectorise,)
    return result


def nothing(tensor):
    """
    Return a tensor

    This is used as a dummy to simplify the backpropagation logic
    """
    return tensor


def select_backend(*tensors: Tensor | np.ndarray | ArrayLike):
    if not CUPY_ENABLED:
        return np

    import cupy

    return cupy.get_array_module(*tensors)
