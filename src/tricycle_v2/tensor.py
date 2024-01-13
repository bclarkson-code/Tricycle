from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

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

        # calculate the gradient for each parameter
        for leaf in leaves.values():
            if leaf.grad_fn is None:
                continue

            for path in leaf.grad_fn:
                grad = np.ones_like(self).view(Tensor)
                grad.requires_grad = False

                for op in path:
                    grad = op(grad)

                leaf.grad = grad if leaf.grad is None else leaf.grad + grad

    def __hash__(self) -> int:
        return id(self)
