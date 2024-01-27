import logging
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

Op = Callable[..., "Tensor"]


class Tensor(np.ndarray):
    """
    An N-dimensional grid of numbers. This is implemented as a subclass
    of a standard numpy array
    """

    _id: int
    args: tuple["Tensor", ...] | None = None
    back_fn: tuple[Op, ...] | None = None
    grad_fn: Optional[List[List[Op]]] = None
    grad: Optional["Tensor"] = None
    name: Optional[str] = None
    requires_grad: bool = False

    def backward(self):
        stack: List[Tuple[Tensor, List[Op]]] = [(self, [])]
        leaves: Dict[int, Tensor] = {}
        # this is for plotting a graph, not needed for autodiff
        adjecency_matrix = defaultdict(list)

        # Find every route to a differentiable parameter
        while stack:
            current_node, current_gradient = stack.pop()

            # At intermediate node
            if current_node.args is None:
                if current_node.grad_fn is None:
                    current_node.grad_fn = [current_gradient]
                else:
                    current_node.grad_fn.append(current_gradient)
                if hash(current_node) not in leaves:
                    leaves[hash(current_node)] = current_node

            else:
                for arg, op in zip(current_node.args, current_node.back_fn):
                    if not arg.requires_grad:
                        continue

                    new_gradient = current_gradient + [op]
                    stack.append((arg, new_gradient))
                    if hash(arg) not in leaves:
                        adjecency_matrix[(hash(current_node))].append(hash(arg))

        # calculate the gradient for each parameter
        for leaf in leaves.values():
            logger.info("------------------finding leaf------------------")
            if leaf.grad_fn is None:
                continue

            for path in leaf.grad_fn:
                grad = np.ones_like(self).view(Tensor)
                grad.requires_grad = False

                logger.info(grad)
                for op in path:
                    grad = op(grad)
                    logger.info(grad)

                leaf.grad = grad if leaf.grad is None else leaf.grad + grad

            logger.info("------------------reached leaf------------------")

        logger.info(adjecency_matrix)
        logger.info(leaves)

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
            raise NotImplementedError(f"Cannot add {type(self)} and {type(other)}")

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
            raise NotImplementedError(f"Cannot sub {type(self)} and {type(other)}")

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
            raise NotImplementedError(f"Cannot mul {type(self)} and {type(other)}")

    def __imul__(self, other):
        return self * other

    def __truediv__(self, other):
        if isinstance(other, np.ndarray) and not isinstance(other, Tensor):
            other = to_tensor(other)
        if np.isscalar(other):
            from tricycle.unary import udiv

            return udiv(other, self)
        elif isinstance(other, Tensor):
            from tricycle.binary import bdiv

            return bdiv(self, other)

        else:
            raise NotImplementedError(f"Cannot divide {type(self)} and {type(other)}")

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


def to_tensor(
    *args, name: Optional[str] = None, requires_grad: bool = True, **kwargs
) -> Tensor:
    """
    Create a new Tensor instance. First, we convert the argument to a numpy
    array and then to a tensor
    """
    result = np.asarray(*args, **kwargs).view(Tensor)
    result.name = name
    result.requires_grad = requires_grad
    return result
