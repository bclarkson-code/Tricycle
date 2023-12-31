import inspect
from functools import partial, wraps
from typing import Callable, Optional, Tuple

import numpy as np

# Type signature for Tensor operation
Op = Callable[..., "Tensor"]

grad: bool = True


def to_tensor(fn: Op) -> Op:
    """
    A decorator to convert non-tensor arguments to tensors
    """

    @wraps(fn)
    def wrapped(*args, **kwargs):
        args = [arg if isinstance(arg, Tensor) else tensor(arg) for arg in args]
        return fn(*args, **kwargs)

    return wrapped


class no_grad:
    """
    A context manager to disable gradient calculation
    """

    def __enter__(self):
        global grad
        grad = False

    def __exit__(self, *_):
        global grad
        grad = True


class Tensor(np.ndarray):
    """
    An N-dimensional grid of numbers. This is implemented as a subclass
    of a standard numpy array
    """

    args: tuple["Tensor", ...] | None = None
    back_fn: tuple[Op, ...] | None = None
    grad_fn: list[Op] = []
    grad: Optional["Tensor"] = None
    name: Optional[str] = None

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
                grad = tensor(np.ones_like(self))
                # if current_node.name == "layer_1_weights":
                #     for fn in current_node.grad_fn:
                #         print(f"-> {fn}")
                #     raise Exception
                for op in current_node.grad_fn:
                    # print(f"-> {op}: {grad}")
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

    def mean(self) -> "Tensor":
        return mean(self)

    @property
    def T(self) -> "Tensor":
        return transpose(self)

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        if self.name is None:
            return f"{super().__repr__()}"
        return f"Tensor({super().__repr__()}, {self.name=})"


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


def tensor(*args, name: Optional[str] = None, **kwargs) -> Tensor:
    """
    Create a new Tensor instance. This is pretty much a copy of the np.array
    constructor that most numpy users use to create an array
    """
    result = np.asarray(*args, **kwargs).view(Tensor)
    result.name = name
    return result


@to_tensor
def sub(x: Tensor, y: Tensor) -> Tensor:
    """
    Subtract two tensors
    """
    global grad
    result = tensor(np.subtract(x, y))
    if grad:
        result.back_fn = (nothing, negate)
        result.args = (x, y)
    return result


@to_tensor
def negate(x: Tensor) -> Tensor:
    """
    Swap the sign of every element of a tensor
    """
    global grad

    result = tensor(np.multiply(x, -1))
    if grad:
        result.back_fn = (negate,)
        result.args = (x,)
    return result


@to_tensor
def nothing(x: Tensor) -> Tensor:
    """
    Do nothing
    """
    global grad
    result = tensor(x)
    if grad:
        result.back_fn = (nothing,)
        result.args = (x,)
    return result


@to_tensor
def add(x: Tensor, y: Tensor) -> Tensor:
    """
    Add two tensors
    """
    global grad
    result = tensor(np.add(x, y))
    if grad:
        result.back_fn = (nothing, nothing)
        result.args = (x, y)
    return result


@to_tensor
def mul(x: Tensor, y: Tensor) -> Tensor:
    """
    Multiply two tensors
    """
    global grad
    result = tensor(np.multiply(x, y))
    if grad:
        result.back_fn = (
            partial(mul, y=y),
            partial(mul, y=x),
        )
        result.args = (x, y)
    return result


@to_tensor
def div(x: Tensor, y: Tensor) -> Tensor:
    """
    Divide two tensors
    """
    global grad
    result = tensor(np.divide(x, y))

    if not grad:
        return result

    def diff_div(arg: Tensor) -> Tensor:
        y_squared = mul(y, y)
        numerator = mul(negate(arg), x)
        return div(numerator, y_squared)

    result.back_fn = (partial(div, y=y), diff_div)
    result.args = (x, y)
    return result


@to_tensor
def reduce_sum(x: Tensor) -> Tensor:
    """
    Sum the elements of a tensor into a single scalar
    """
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    indices = alphabet[: len(x.shape)]
    subscripts = f"{indices}->"
    return einsum(x, subscripts=subscripts)


@to_tensor
def pow(x: Tensor, y: Tensor) -> Tensor:
    """
    Raise every element of a tensor to the power of another tensor
    """
    global grad
    result = tensor(np.power(x, y))
    if not grad:
        return result

    def diff_power_arg_1(arg: Tensor) -> Tensor:
        return (pow(x, y - 1) * y) * arg

    def diff_power_arg_2(arg: Tensor) -> Tensor:
        return pow(x, y) * log(x) * arg

    result.back_fn = (
        diff_power_arg_1,
        diff_power_arg_2,
    )
    result.args = (x, y)
    return result


@to_tensor
def exp(x: Tensor) -> Tensor:
    """
    Raise every element of a tensor to the power of e
    """
    global grad
    result = tensor(np.exp(x))
    if not grad:
        return result
    result.back_fn = (exp,)
    result.args = (x,)
    return result


@to_tensor
def log(x: Tensor) -> Tensor:
    """
    Find the natural log of every element of a tensor
    """
    global grad
    result = tensor(np.log(x))
    if not grad:
        return result

    result.back_fn = (bind(div, 1, ...),)
    result.args = (x,)
    return result


@to_tensor
def sqrt(x: Tensor) -> Tensor:
    """
    Find the square root of every element of a tensor
    """
    global grad
    result = tensor(np.sqrt(x))
    if not grad:
        return result

    def diff_sqrt(arg: Tensor) -> Tensor:
        return pow(arg, -0.5) * 0.5

    result.back_fn = (diff_sqrt,)
    result.args = (x,)
    return result


@to_tensor
def sin(x: Tensor) -> Tensor:
    """
    Find the sine of every element of a tensor
    """
    global grad
    result = tensor(np.sin(x))
    if not grad:
        return result

    result.back_fn = (cos,)
    result.args = (x,)
    return result


@to_tensor
def cos(x: Tensor) -> Tensor:
    """
    Find the cosine of every element of a tensor
    """
    global grad
    result = tensor(np.cos(x))
    if not grad:
        return result

    def diff_cos(arg: Tensor) -> Tensor:
        return -sin(arg)

    result.back_fn = (diff_cos,)
    result.args = (x,)
    return result


@to_tensor
def max(x: Tensor) -> Tensor:
    """
    Find the largest element of a tensor
    """
    global grad
    result = tensor(x.max())
    if not grad:
        return result

    def diff_max(arg: Tensor) -> Tensor:
        return tensor(arg == x.max())

    result.back_fn = (diff_max,)
    result.args = (x,)
    return result


@to_tensor
def min(x: Tensor) -> Tensor:
    """
    Find the smallest element of a tensor
    """
    global grad
    result = tensor(x.min())
    if not grad:
        return result

    def diff_min(arg: Tensor) -> Tensor:
        return tensor(arg == x.min())

    result.back_fn = (diff_min,)
    result.args = (x,)
    return result


def _parse_subscripts(subscripts: str) -> tuple[list[str], str]:
    indices, result = subscripts.split("->")
    indices = indices.split(",")
    return indices, result


def _to_binary(tensors: list[Tensor], subscripts: str) -> tuple[list[Tensor], str]:
    """
    If a singular operation is passed, (e.g add along an index)
    we need some way o figuring out how much to expand in the back operation
    We can do this by converting the singular operation to a binary operation
    by multiplying elementwise by a matrix of ones
    """
    assert len(tensors) == 1, "Operation is already binary"
    [index], result = _parse_subscripts(subscripts)
    if not result:
        result = "..."
    subscripts = f"{index},{index}->{result}"

    tensors = (tensor(np.ones_like(tensors[0])), tensors[0])
    return tensors, subscripts


@to_tensor
def einsum(*tensors: Tensor, subscripts: str) -> Tensor:
    """
    Use einstein summation notation to perform tensor operations on
    some tensors"""
    global grad
    if len(tensors) == 1:
        tensors, subscripts = _to_binary(tensors, subscripts)

    result = tensor(np.einsum(subscripts, *tensors))

    if not grad:
        return result

    back_fns = []


    indices, output = _parse_subscripts(subscripts)
    if not output:
        output = "..."


    for idx in range(len(tensors)):
        left = tensors[:idx]
        right = tensors[idx + 1 :]

        indices, output = _parse_subscripts(subscripts)
        output, indices[idx] = indices[idx], output
        indices = ",".join(indices)
        diff_subscripts = f"{indices}->{output}"

        def diff_einsum(
            arg: Tensor, left=left, right=right, subscripts=diff_subscripts
        ) -> Tensor:
            """
            Derivative of einsum wrt a single input tensor
            """
            args = left + (arg,) + right
            return einsum(*args, subscripts=subscripts)

        back_fns.append(diff_einsum)

    result.args = tuple(tensors)
    result.back_fn = tuple(back_fns)

    return result


@to_tensor
def matmul(x: Tensor, y: Tensor) -> Tensor:
    """
    Compute the matrix multiplication of two tensors
    """
    alphabet = "abcdefghjklmnopqrstuvwxyz"
    assert len(x.shape) + len(y.shape) <= len(
        alphabet
    ), "Cannot perform matmul on tensors with more than 26 dimensions in total"
    assert len(x.shape), "Cannot perform matmul on a 0D tensor"
    assert len(y.shape), "Cannot perform matmul on a 0D tensor"

    left_indices = alphabet[: len(x.shape) - 1] if len(x.shape) > 1 else ""
    right_indices = alphabet[-len(y.shape) + 1 :] if len(y.shape) > 1 else ""
    subscripts = f"{left_indices}i,i{right_indices}->{left_indices}{right_indices}"
    return einsum(x, y, subscripts=subscripts)


@to_tensor
def transpose(x: Tensor) -> Tensor:
    """
    Compute the transpose of a tensor
    """
    assert (
        len(x.shape) <= 2
    ), "Can only perform transpose on 2D tensors. Use einsum for more complex cases."
    return einsum(x, subscripts="ij->ji")


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


@to_tensor
def relu(x: Tensor) -> Tensor:
    """
    Compute the relu of a tensor
    """
    result = tensor(np.maximum(x, 0))

    def diff_relu(arg: Tensor) -> Tensor:
        weight = (x > 0).astype(x.dtype)
        return einsum(weight, arg, subscripts="ij,ij->ij")

    result.back_fn = (diff_relu,)
    result.args = (x,)

    return result
