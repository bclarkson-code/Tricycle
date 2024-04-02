import gc
import logging
import numbers
import uuid
from copy import copy
from typing import Callable, List, Optional, Sequence, Union

import numpy as np
from numpy.typing import ArrayLike

logger = logging.getLogger(__name__)

Op = Callable[..., "Tensor"]


class Tensor:
    """
    An N-dimensional grid of numbers. This is implemented as a subclass
    of a standard numpy array
    """

    _id: int
    _grad_fn: Optional[List[List[Op]]] = None
    _data: ArrayLike
    args: tuple["Tensor", ...] | None = None
    back_fns: tuple[Op, ...] | None = None
    parents: set["Tensor"] | None = None
    grad: Optional["Tensor"] = None
    name: Optional[str] = None
    requires_grad: bool = False
    is_vector: bool = False

    def __init__(
        self,
        data: ArrayLike,
        requires_grad: bool = False,
        is_vector: bool = False,
        name: str | None = None,
        _id: int | None = None,
    ):
        self._id = _id or uuid.uuid4().int
        self._data = data
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
                    arg.parents = set()

                if node not in arg.parents:
                    stack.append(arg)
                    arg.parents.add(node)

    def _calculate_gradients(self):
        """
        Traverse through the graph, calculating gradients along the way such
        that a child is only visited if the entirety of its parent's gradient
        has been computed
        """
        self.grad = to_tensor(
            np.ones_like(self),
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
                if arg.grad is None:
                    arg.grad = back_fns(node.grad)
                else:
                    arg.grad += back_fns(node.grad)

                # only move to arg if we have been to all of its parents
                if len(arg.parents) == 0:
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

            # delete node
            if hasattr(node, "grad") and node.grad is not None:
                del node.grad
            del node
        gc.collect()

    def backward(self):
        """
        Perform a backward pass through the graph, calculating the gradient
        for each parameter
        """
        self._attach_parents()
        self._calculate_gradients()

    def __hash__(self) -> int:
        return id(self)

    def __add__(self, other: Union[float, "Tensor"]) -> "Tensor":
        if isinstance(other, numbers.Number):
            from tricycle.unary import uadd

            return uadd(self, other)
        elif isinstance(other, Tensor):
            from tricycle.binary import badd

            return badd(self, other)
        else:
            raise NotImplementedError(
                f"Cannot add {type(self)} and {type(other)}"
            )

    def __radd__(self, other):
        return self + other

    def __iadd__(self, other):
        return self + other

    def __sub__(self, other):
        if isinstance(other, np.ndarray) and not isinstance(other, Tensor):
            other = to_tensor(other)
        if np.isscalar(other):
            from tricycle.unary import usub

            return usub(self, other)
        elif isinstance(other, Tensor):
            from tricycle.binary import bsub

            return bsub(self, other)

        else:
            raise NotImplementedError(
                f"Cannot sub {type(self)} and {type(other)}"
            )

    def __rsub__(self, other):
        return -(self - other)

    def __isub__(self, other):
        return self.__sub__(other)

    def __mul__(self, other):
        if isinstance(other, np.ndarray) and not isinstance(other, Tensor):
            other = to_tensor(other)
        if np.isscalar(other) or other.shape == ():
            from tricycle.unary import umul

            return umul(self, other)

        elif isinstance(other, Tensor):
            from tricycle.binary import bmul

            return bmul(self, other)

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
        if np.isscalar(other):
            from tricycle.unary import umul

            return umul(self, 1 / other)
        elif isinstance(other, Tensor):
            from tricycle.binary import bdiv

            return bdiv(self, other)

        else:
            raise NotImplementedError(
                f"Cannot divide {type(self)} and {type(other)}"
            )

    def __itruediv__(self, other):
        return self / other

    def __floordiv__(self, _):
        raise NotImplementedError("Cannot floor divide")

    def __mod__(self, _):
        raise NotImplementedError("Cannot mod")

    def __pow__(self, other) -> "Tensor":
        if isinstance(other, np.ndarray) and not isinstance(other, Tensor):
            other = to_tensor(other)
        if np.isscalar(other):
            from tricycle.unary import upow

            return upow(self, other)
        elif isinstance(other, Tensor):
            raise NotImplementedError(
                f"Cannot power two tensors of shape: {self.shape}, {other.shape}"
            )
        else:
            raise NotImplementedError(
                f"Cannot power {type(self)} and {type(other)}"
            )

    def __repr__(self):
        name = f", name={self.name}" if self.name is not None else ""
        return f"Tensor({self._data.__str__()}{name})"

    def e(self, subscript: str) -> "Tensor":
        from tricycle.einsum import Einsum

        return Einsum(subscript)(self)

    def repeat(self, n_repeats: int) -> "Tensor":
        from tricycle.ops import repeat

        return repeat(self, n_repeats)

    @property
    def shape(self) -> Sequence[int]:
        return self._data.shape

    def ndim(self) -> int:
        return self._data.ndim

    def reshape(self, shape: Sequence[int]) -> "Tensor":
        from tricycle.ops import reshape

        return reshape(self, shape)

    def split(self, n_splits: int, axis: int = 0) -> List["Tensor"]:
        from tricycle.ops import split

        return split(self, n_splits, axis)

    def mean(self) -> "Tensor":
        from tricycle.ops import mean

        return mean(self)

    def variance(self) -> "Tensor":
        from tricycle.ops import variance

        return variance(self)

    def standard_deviation(self) -> "Tensor":
        from tricycle.ops import standard_deviation

        return standard_deviation(self)

    def normalise(self) -> "Tensor":
        from tricycle.ops import normalise

        return normalise(self)

    def close_to(
        self, other: "Tensor" | ArrayLike, equal_nan=False, **kwargs
    ) -> bool:
        """
        Convenience method to check if two tensors are identical
        to within some tolerance
        """
        return np.allclose(
            np.array(self), np.array(other), equal_nan=equal_nan, **kwargs
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

    def numpy(self):
        return np.array(self)


def to_tensor(
    *args,
    name: Optional[str] = None,
    requires_grad: bool = True,
    is_vector: bool = False,
    _id: int | None = None,
    **kwargs,
) -> Tensor:
    """
    Create a new Tensor instance. First, we convert the argument to a numpy
    array and then to a tensor
    """
    return Tensor(
        np.asarray(*args, **kwargs),
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
        raise ValueError("Tensor is already vectorised")

    result = to_tensor(
        copy(tensor), is_vector=True, requires_grad=tensor.requires_grad
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
        raise ValueError("Tensor is not vectorised")

    result = to_tensor(
        copy(tensor), is_vector=False, requires_grad=tensor.requires_grad
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
