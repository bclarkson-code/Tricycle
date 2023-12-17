from functools import partial
from typing import Callable, Optional

import numpy as np

# Type signature for Tensor operation
Op = Callable[..., "Tensor"]


def to_tensor(fn: Op) -> Op:
    """
    A decorator to convert non-tensor arguments to tensors
    """

    def wrapped(*args, **kwargs):
        args = [arg if isinstance(arg, Tensor) else tensor(arg) for arg in args]
        return fn(*args, **kwargs)

    return wrapped


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
                grad = tensor(np.ones_like(current_node))
                for op in current_node.grad_fn[::-1]:
                    # Actually calculate the gradient for a node
                    grad = op(grad)

                if current_node.grad is None:
                    current_node.grad = grad
                else:
                    # If there are multiple paths to this node, we
                    # need to add all of the gradients together
                    current_node.grad += grad

    @to_tensor
    def __add__(self, other: "Tensor") -> "Tensor":
        return add(self, other)

    @to_tensor
    def __sub__(self, other: "Tensor") -> "Tensor":
        return sub(self, other)

    @to_tensor
    def __mul__(self, other: "Tensor") -> "Tensor":
        return mul(self, other)

    @to_tensor
    def __truediv__(self, other: "Tensor") -> "Tensor":
        return div(self, other)

    @to_tensor
    def __pow__(self, other: "Tensor") -> "Tensor":
        return pow(self, other)

    @to_tensor
    def __neg__(self) -> "Tensor":
        return negate(self)

    @to_tensor
    def __matmul__(self, other: "Tensor") -> "Tensor":
        return matmul(self, other)

    @to_tensor
    def mean(self) -> "Tensor":
        return mean(self)


class bind(partial):
    """
    A version of partial which accepts Ellipsis (...) as a placeholder
    credit: https://stackoverflow.com/questions/7811247/
    """

    def __call__(self, *args, **keywords):
        keywords = {**self.keywords, **keywords}
        iargs = iter(args)
        args = (next(iargs) if arg is ... else arg for arg in self.args)
        return self.func(*args, *iargs, **keywords)


def tensor(*args, **kwargs):
    """
    Create a new Tensor instance. This is pretty much a copy of the np.array
    constructor that most numpy users use to create an array
    """
    return np.asarray(*args, **kwargs).view(Tensor)


def to_tensor(fn: Op) -> Op:
    """
    A decorator to convert non-tensor arguments to tensors
    """

    def wrapped(*args, **kwargs):
        args = [arg if isinstance(arg, Tensor) else tensor(arg) for arg in args]
        return fn(*args, **kwargs)

    return wrapped


def _no_grad(fn: Op) -> Op:
    """
    A helper function to mark an operation as not requiring gradient
    calculations
    """
    return partial(fn, grad=False)


@to_tensor
def sub(x: Tensor, y: Tensor, grad=True) -> Tensor:
    """
    Subtract two tensors
    """
    result = tensor(np.subtract(x, y))
    if grad:
        result.back_fn = (_no_grad(nothing), _no_grad(negate))
        result.args = (x, y)
    return result


@to_tensor
def negate(x: Tensor, grad=True) -> Tensor:
    """
    Swap the sign of every element of a tensor
    """
    result = tensor(np.multiply(x, -1))
    if grad:
        result.back_fn = (_no_grad(negate),)
        result.args = (x,)
    return result


@to_tensor
def nothing(x: Tensor, grad=True) -> Tensor:
    """
    Do nothing
    """
    result = tensor(x)
    if grad:
        result.back_fn = (_no_grad(nothing),)
        result.args = (x,)
    return result


@to_tensor
def add(x: Tensor, y: Tensor, grad=True) -> Tensor:
    """
    Add two tensors
    """
    result = tensor(np.add(x, y))
    if grad:
        result.back_fn = (_no_grad(nothing), _no_grad(nothing))
        result.args = (x, y)
    return result


@to_tensor
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


@to_tensor
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


@to_tensor
def reduce_sum(x: Tensor, grad=True) -> Tensor:
    """
    Sum the elements of a tensor into a single scalar
    """
    result = tensor(np.sum(x))
    if grad:
        result.back_fn = (_no_grad(reduce_sum),)
        result.args = (x,)
    return result


@to_tensor
def pow(x: Tensor, y: Tensor, grad=True) -> Tensor:
    """
    Raise every element of a tensor to the power of another tensor
    """
    result = tensor(np.power(x, y))
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


@to_tensor
def exp(x: Tensor, grad=True) -> Tensor:
    """
    Raise every element of a tensor to the power of e
    """
    result = tensor(np.exp(x))
    if not grad:
        return result
    result.back_fn = (_no_grad(exp),)
    result.args = (x,)
    return result


@to_tensor
def log(x: Tensor, grad=True) -> Tensor:
    """
    Find the natural log of every element of a tensor
    """
    result = tensor(np.log(x))
    if not grad:
        return result

    result.back_fn = (bind(_no_grad(div), 1, ...),)
    result.args = (x,)
    return result


@to_tensor
def sqrt(x: Tensor, grad=True) -> Tensor:
    """
    Find the square root of every element of a tensor
    """
    result = tensor(np.sqrt(x))
    if not grad:
        return result

    def diff_sqrt(arg: Tensor) -> Tensor:
        no_grad_pow = _no_grad(pow)
        no_grad_mul = _no_grad(mul)

        POWER = tensor(-1 / 2)
        ONE_HALF = tensor(1 / 2)
        return no_grad_mul(no_grad_pow(arg, POWER), ONE_HALF)

    result.back_fn = (diff_sqrt,)
    result.args = (x,)
    return result


@to_tensor
def sin(x: Tensor, grad=True) -> Tensor:
    """
    Find the sine of every element of a tensor
    """
    result = tensor(np.sin(x))
    if not grad:
        return result

    result.back_fn = (_no_grad(cos),)
    result.args = (x,)
    return result


@to_tensor
def cos(x: Tensor, grad=True) -> Tensor:
    """
    Find the cosine of every element of a tensor
    """
    result = tensor(np.cos(x))
    if not grad:
        return result

    def diff_cos(arg: Tensor) -> Tensor:
        no_grad_sin = _no_grad(sin)
        no_grad_negate = _no_grad(negate)
        return no_grad_negate(no_grad_sin(arg))

    result.back_fn = (diff_cos,)
    result.args = (x,)
    return result


@to_tensor
def max(x: Tensor, grad=True) -> Tensor:
    """
    Find the largest element of a tensor
    """
    result = tensor(x.max())
    if not grad:
        return result

    def diff_max(arg: Tensor) -> Tensor:
        return tensor(arg == x.max())

    result.back_fn = (diff_max,)
    result.args = (x,)
    return result


@to_tensor
def min(x: Tensor, grad=True) -> Tensor:
    """
    Find the smallest element of a tensor
    """
    result = tensor(x.min())
    if not grad:
        return result

    def diff_min(arg: Tensor) -> Tensor:
        return tensor(arg == x.min())

    result.back_fn = (diff_min,)
    result.args = (x,)
    return result


@to_tensor
def matmul(x: Tensor, y: Tensor, grad=True) -> Tensor:
    """
    Compute the matrix multiplication of two tensors
    """
    result = tensor(np.matmul(x, y))

    if not grad:
        return result

    result.back_fn = (
        partial(_no_grad(matmul), y=y),
        partial(_no_grad(matmul), y=x),
    )
    result.args = (x, y)
    return result


@to_tensor
def mean(x: Tensor):
    """
    Compute the mean of a tensor
    """
    return reduce_sum(x) / x.shape[0]


@to_tensor
def softmax(x: Tensor):
    """
    Compute the softmax of a tensor
    """
    return exp(x) / reduce_sum(exp(x))


@to_tensor
def sigmoid(x: Tensor) -> Tensor:
    """
    Compute the sigmoid of a tensor
    """
    return tensor(1) / (exp(-x) + 1)
