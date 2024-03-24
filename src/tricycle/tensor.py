import logging
import uuid
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import ArrayLike

logger = logging.getLogger(__name__)

Op = Callable[..., "Tensor"]


class Tensor(np.ndarray):
    """
    An N-dimensional grid of numbers. This is implemented as a subclass
    of a standard numpy array
    """

    _id: int
    _grad_fn: Optional[List[List[Op]]] = None
    args: tuple["Tensor", ...] | None = None
    back_fn: tuple[Op, ...] | None = None
    grad: Optional["Tensor"] = None
    name: Optional[str] = None
    requires_grad: bool = False
    is_vector: bool = False

    def _find_differentiable_params(self) -> Dict[int, "Tensor"]:
        """
        Find every path backward through the computational graph from the current tensor
        to every differentiable parameter and attach them to the corresponding
        differentiable_params
        """
        stack: List[Tuple[Tensor, List[Op]]] = [(self, [])]
        differentiable_params: Dict[int, Tensor] = {}

        # Find every route to a differentiable parameter
        while stack:
            current_node, current_gradient = stack.pop()

            # At leaf node
            if current_node.args is None:
                if current_node._grad_fn is None:
                    current_node._grad_fn = [current_gradient]
                else:
                    current_node._grad_fn.append(current_gradient)
                if hash(current_node) not in differentiable_params:
                    differentiable_params[hash(current_node)] = current_node

            # At non-leaf node
            else:
                for arg, op in zip(current_node.args, current_node.back_fn):
                    if not arg.requires_grad:
                        continue

                    new_gradient = current_gradient + [op]
                    stack.append((arg, new_gradient))
        return differentiable_params

    def _calculate_gradient(self, param: "Tensor") -> None:
        """
        Calculate the gradient for a single parameter in the computational
        graph
        """
        if param._grad_fn is None:
            return

        for path in param._grad_fn:
            grad = to_tensor(
                np.ones_like(self),
                requires_grad=False,
                is_vector=self.is_vector,
            )

            logger.debug(f"  {path}")
            logger.debug("----------------------------")
            for op in path:
                logger.debug(f"  {op}")
                grad = op(grad)
                logger.debug(f"  {grad.shape}")
                logger.debug(f"  {grad.is_vector}")
            logger.debug("----------------------------")

            param.grad = grad if param.grad is None else param.grad + grad
        param._grad_fn = None

    def backward(self):
        """
        Perform a backward pass through the graph, calculating the gradient
        for each parameter
        """
        params = self._find_differentiable_params()
        for param in params.values():
            self._calculate_gradient(param)

    def __hash__(self) -> int:
        return id(self)

    def __add__(self, other):
        if isinstance(other, np.ndarray) and not isinstance(other, Tensor):
            other = to_tensor(other)
        if np.isscalar(other):
            from tricycle.unary import uadd

            return uadd(self, other)
        elif isinstance(other, Tensor):
            from tricycle.binary import badd

            return badd(self, other)
        else:
            raise NotImplementedError(
                f"Cannot add {type(self)} and {type(other)}"
            )

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

    def __isub__(self, other):
        return -1 * (other - self)

    def __mul__(self, other):
        if isinstance(other, np.ndarray) and not isinstance(other, Tensor):
            other = to_tensor(other)
        if np.isscalar(other):
            from tricycle.unary import umul

            return umul(self, other)

        elif isinstance(other, Tensor):
            from tricycle.binary import bmul

            return bmul(self, other)

        else:
            raise NotImplementedError(
                f"Cannot mul {type(self)} and {type(other)}"
            )

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

    def __pow__(self, other):
        if isinstance(other, np.ndarray) and not isinstance(other, Tensor):
            other = to_tensor(other)
        if np.isscalar(other):
            from tricycle.unary import upow

            return upow(self, other)
        elif isinstance(other, Tensor):
            raise NotImplementedError("Cannot power")

    def __repr__(self):
        name = f", name={self.name}" if self.name is not None else ""
        return f"Tensor({self.__str__()}{name})"

    def __new__(
        cls,
        shape,
        dtype=float,
        buffer=None,
        offset=0,
        strides=None,
        order=None,
    ):
        obj = super().__new__(
            cls, shape, dtype, buffer, offset, strides, order
        )
        obj.uuid = uuid.uuid4()
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.uuid = getattr(obj, "uuid", None)

    def e(self, subscript: str) -> "Tensor":
        from tricycle.einsum import Einsum

        return Einsum(subscript)(self)

    def repeat(self, n_repeats: int) -> "Tensor":
        from tricycle.ops import repeat

        return repeat(self, n_repeats)

    def reshape(self, shape: Sequence[int]) -> "Tensor":
        from tricycle.ops import reshape

        return reshape(self, shape)

    def split(self, n_splits: int, axis: int = 0) -> List["Tensor"]:
        from tricycle.ops import split

        return split(self, n_splits, axis)

    def close_to(self, other: "Tensor" | ArrayLike, **kwargs) -> bool:
        """
        Convenience method to check if two tensors are identical
        to within some tolerance
        """
        return np.allclose(np.array(self), np.array(other), **kwargs)

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
    **kwargs,
) -> Tensor:
    """
    Create a new Tensor instance. First, we convert the argument to a numpy
    array and then to a tensor
    """
    result = np.asarray(*args, **kwargs).view(Tensor)
    result.name = name
    result.requires_grad = requires_grad
    result.uuid = uuid.uuid4()
    result.is_vector = is_vector
    return result


def vectorise(tensor: Tensor) -> Tensor:
    """
    Tell Tricycle to treat this tensor as a group of vectors
    """
    if tensor.is_vector:
        raise ValueError("Tensor is already vectorised")

    result = to_tensor(tensor, is_vector=True)
    result.args = (tensor,)
    result.back_fn = (unvectorise,)
    return result


def unvectorise(tensor: Tensor) -> Tensor:
    """
    Tell Tricycle to treat this tensor as a single tensor
    (not a group of vectors)
    """
    if not tensor.is_vector:
        raise ValueError("Tensor is not vectorised")

    result = to_tensor(tensor, is_vector=False)
    result.args = (tensor,)
    result.back_fn = (vectorise,)
    return result
