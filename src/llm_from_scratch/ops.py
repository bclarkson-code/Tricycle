from functools import partial
from typing import Callable, Optional

import numpy as np

# Type signature for Tensor operation
Op = Callable[..., "Tensor"]


class Tensor(np.ndarray):
    """
    An N-dimensional grid of numbers. This is implemented as a subclass
    of a standard numpy array
    """

    args: tuple["Tensor", ...] | None = None
    back_fn: tuple[Op, ...] | None = None
    grad_fn: list[Op] = []
    grad: Optional["Tensor"] = None
    requires_grad: bool = True

    def backward(self):
        """
        Perform auto-differentiation by traversing the computational graph
        using a depth first search, calculating the gradient along the way
        """

        stack = [self]

        while stack:
            current_node = stack.pop()

            # At intermediate node
            if (
                current_node.back_fn is not None
                and current_node.args is not None  # noqa: E501
            ):
                # Get gradient functions for the operation
                grad_fn = current_node.grad_fn

                # Update gradient functions for each parent node
                for arg, op in zip(current_node.args, current_node.back_fn):
                    arg.grad_fn = grad_fn + [op]

                    # Add each arg to the stack for further processing
                    stack.append(arg)  # type: ignore

            # At leaf node
            else:
                grad = tensor(1)
                for op in current_node.grad_fn[::-1]:
                    # Actually calculate the gradient for a node
                    grad = op(grad)

                if current_node.grad is None:
                    current_node.grad = grad
                else:
                    # If there are multiple paths to this node, we
                    # need to add all of the gradients together
                    current_node.grad += grad


def tensor(*args, **kwargs):
    """
    Create a new Tensor instance. This is pretty much a copy of the np.array
    constructor that most numpy users use to create an array
    """
    return np.asarray(*args, **kwargs).view(Tensor)


def _no_grad(fn: Op) -> Op:
    """
    A helper function to mark an operation as not requiring gradient
    calculations
    """
    return partial(fn, grad=False)


def sub(x: Tensor, y: Tensor, grad=True) -> Tensor:
    """
    Subtract two tensors
    """
    result = tensor(np.subtract(x, y))
    if grad:
        result.back_fn = (_no_grad(nothing), _no_grad(negate))
        result.args = (x, y)
    return result


def negate(x: Tensor, grad=True) -> Tensor:
    """
    Swap the sign of every element of a tensor
    """
    result = tensor(np.multiply(x, -1))
    if grad:
        result.back_fn = (_no_grad(negate),)
        result.args = (x,)
    return result


def nothing(x: Tensor, grad=True) -> Tensor:
    """
    Do nothing
    """
    result = tensor(x)
    if grad:
        result.back_fn = (_no_grad(nothing),)
        result.args = (x,)
    return result


def add(x: Tensor, y: Tensor, grad=True) -> Tensor:
    """
    Add two tensors
    """
    result = tensor(np.add(x, y))
    if grad:
        result.back_fn = (_no_grad(nothing), _no_grad(nothing))
        result.args = (x, y)
    return result


def mul(x: Tensor, y: Tensor, grad=True) -> Tensor:
    """
    Multiply two tensors
    """
    result = tensor(np.multiply(x, y))
    if grad:
        result.back_fn = (
            partial(_no_grad(mul), y=y),
            partial(_no_grad(mul), y=x),
        )
        result.args = (x, y)
    return result


def div(x: Tensor, y: Tensor, grad=True) -> Tensor:
    """
    Divide two tensors
    """
    result = tensor(np.divide(x, y))

    if not grad:
        return result

    def diff_div(arg: Tensor) -> Tensor:
        no_grad_mul = _no_grad(mul)
        no_grad_negate = _no_grad(negate)
        no_grad_div = _no_grad(div)

        y_squared = no_grad_mul(y, y)
        numerator = no_grad_mul(no_grad_negate(arg), x)
        return no_grad_div(numerator, y_squared)

    result.back_fn = (partial(_no_grad(div), y=y), diff_div)
    result.args = (x, y)
    return result


def reduce_sum(x: Tensor, grad=True) -> Tensor:
    """
    Sum the elements of a tensor into a single scalar
    """
    result = tensor(x.sum())
    if grad:
        result.back_fn = (_no_grad(reduce_sum),)
        result.args = (x,)
    return result


def pow(x: Tensor, y: Tensor, grad=True) -> Tensor:
    """
    Raise every element of a tensor to the power of another tensor
    """
    result = tensor(np.power(x, y))
    """
    case 0:
        return lambda grad: mul_(
            grad, self.y.value * pow_(self.x.value, self.y.value - 1)
        )
    case 1:
        return lambda grad: mul_(
            grad, np.log(self.x.value) * pow_(self.x.value, self.y.value)
        )

    """
    if not grad:
        return result

    def diff_power_arg_1(arg: Tensor) -> Tensor:
        no_grad_pow = _no_grad(pow)
        no_grad_mul = _no_grad(mul)
        no_grad_sub = _no_grad(sub)
        ONE = tensor(1)

        coef = no_grad_pow(x, no_grad_sub(y, ONE))
        coef = no_grad_mul(coef, y)
        result = no_grad_mul(arg, coef)

        return result

    def diff_power_arg_2(arg: Tensor) -> Tensor:
        no_grad_pow = _no_grad(pow)
        no_grad_mul = _no_grad(mul)
        no_grad_log = _no_grad(log)

        coef = no_grad_pow(x, y)
        coef = no_grad_mul(coef, no_grad_log(x))
        result = no_grad_mul(arg, coef)

        return result

    result.back_fn = (
        diff_power_arg_1,
        diff_power_arg_2,
    )
    result.args = (x, y)
    return result


def exp(x: Tensor, grad=True) -> Tensor:
    """
    Raise every element of a tensor to the power of e
    """
    return np.exp(x)


def log(x: Tensor, grad=True) -> Tensor:
    """
    Find the natural log of every element of a tensor
    """
    return np.log(x)


def sqrt(x: Tensor, grad=True) -> Tensor:
    """
    Find the square root of every element of a tensor
    """
    return np.sqrt(x)


def sin(x: Tensor, grad=True) -> Tensor:
    """
    Find the sine of every element of a tensor
    """
    return np.sin(x)


def cos(x: Tensor, grad=True) -> Tensor:
    """
    Find the cosine of every element of a tensor
    """
    return np.cos(x)


def max(x: Tensor, grad=True) -> Tensor:
    """
    Find the largest element of a tensor
    """
    return x.max()


def min(x: Tensor, grad=True) -> Tensor:
    """
    Find the smallest element of a tensor
    """
    return x.min()


def dot(x: Tensor, y: Tensor, grad=True) -> Tensor:
    """
    Compute the dot product of two tensors
    """
    return np.dot(x, y)
