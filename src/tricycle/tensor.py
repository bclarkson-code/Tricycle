import logging
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout

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
    show_graph = False

    def _find_differentiable_params(self) -> Dict[int, "Tensor"]:
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
            grad = np.ones_like(self).view(Tensor)
            grad.requires_grad = False

            for op in path:
                grad = op(grad)

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
        if np.isscalar(other):
            from tricycle.unary import umul

            return umul(self, 1 / other)
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
