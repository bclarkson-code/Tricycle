from functools import partial, wraps
from string import ascii_letters
from typing import Callable, Optional

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
        to find every path from the root (loss) to each leaf (parameter).

        Once we reach a leaf, we calculate its gradient by applying
        the gradient function for each operation that we passed along the
        path
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
                for op in current_node.grad_fn:
                    # Follow the path from root to leaf, applying
                    # each function to get the gradient for the leaf
                    #
                    # TODO: this is slow
                    #
                    # - We can speed this up by caching intermediate values
                    # that are used by multiple leaves
                    #
                    # - We should also be able to speed this up by analysing
                    # the path to see if any operations can be fused
                    #
                    # - We should also probably ignore leaves that we don't
                    # need the gradient for (e.g the inputs)
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
    def __iadd__(self, other: "Tensor") -> "Tensor":
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
    Create a new Tensor instance. First, we convert the argument to a numpy
    array and then to a tensor
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
    indices = ascii_letters[: len(x.shape)]
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
    """
    Parse a subscripts string into a list of indices and a result
    """
    indices, result = subscripts.split("->")
    indices = indices.split(",")
    return indices, result


@to_tensor
def einsum(*tensors: Tensor, subscripts: str) -> Tensor:
    """
    The way we'll be doing tensor manipulation is with the einsum function from
    numpy. This uses a version of einstein summation notation to do tensor
    operations. It takes a bit of getting used to but ends up being a really
    powerful way to do tensor operations. It feels like the natural way to
    do deep learning

    Examples
    --------
    >>> a = np.arange(25).reshape(5,5)

    Trace of a matrix:

    >>> np.einsum('ii', a)
    60

    Sum over an axis:

    >>> np.einsum('ij->i', a)
    array([ 10,  35,  60,  85, 110])

    Tensor contraction:

    >>> a = np.arange(60.).reshape(3,4,5)
    >>> b = np.arange(24.).reshape(4,3,2)
    >>> np.einsum('ijk,jil->kl', a, b)
    array([[4400., 4730.],
           [4532., 4874.],
           [4664., 5018.],
           [4796., 5162.],
           [4928., 5306.]])

    """
    global grad

    # We need to handle ops that have a single input or single value output
    # carefully
    indices, output = _parse_subscripts(subscripts)

    if len(tensors) == 1:
        indices = [indices[0], indices[0]]
        tensors = (tensor(np.ones_like(tensors[0])), tensors[0])

    if output == "":
        output = "..."

    indices = ",".join(indices)
    subscripts = f"{indices}->{output}"

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
    # We are limited by the number of available letters but thankfully
    # numpy has a max dimension of 32 so this shouldn't be too much of a
    # problem. If we start doing giant einsums then i'll probably need to
    # build a custom implementation of einsum but we should be good for now
    assert len(x.shape) + len(y.shape) <= len(
        ascii_letters
    ), f"Cannot perform matmul on tensors with more than  {len(ascii_letters)} dimensions in total"
    assert len(x.shape), "Cannot perform matmul on a 0D tensor"
    assert len(y.shape), "Cannot perform matmul on a 0D tensor"

    left_indices = ascii_letters[: len(x.shape) - 1] if len(x.shape) > 1 else ""
    right_indices = ascii_letters[-len(y.shape) + 1 :] if len(y.shape) > 1 else ""

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
    # For numeric stability we'll subtract the max of the tensor
    x = exp(x - max(x))
    assert len(x.shape) < len(
        ascii_letters
    ), f"Cannot perform softmax on tensors with more than {len(ascii_letters)} dimensions"


    if len(x.shape) == 1:
        denom = 1 / einsum(x, subscripts="i->")
        return einsum(denom, x, subscripts=",i->i")

    indices = ascii_letters[: len(x.shape) - 1]
    final_letter = ascii_letters[-1]
    denom = 1 / einsum(x, subscripts=f"{final_letter}{indices}->{final_letter}")
    return einsum(
        denom,
        x,
        subscripts=f"{final_letter},{final_letter}{indices}->{final_letter}{indices}",
    )


@to_tensor
def sigmoid(x: Tensor) -> Tensor:
    """
    Compute the sigmoid of a tensor
    """
    return tensor(1) / (exp(-x) + 1)
