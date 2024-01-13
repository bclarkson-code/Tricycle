import logging
import uuid
from functools import partial, wraps
from string import ascii_letters, ascii_lowercase
from typing import Callable, List, Optional, Tuple, Union

import networkx as nx
import numpy as np
import pydot
from matplotlib import pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout

# Type signature for Tensor operation
Op = Callable[..., "Tensor"]

grad: bool = True

logger = logging.getLogger(__name__)


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

    _id: int
    args: tuple["Tensor", ...] | None = None
    back_fn: tuple[Op, ...] | None = None
    grad_fn: List[List[Op]] = []
    grad: Optional["Tensor"] = None
    name: Optional[str] = None
    requires_grad: bool = False

    def backward(self):
        """
        Perform auto-differentiation by traversing the computational graph
        to find every path from the root (loss) to each leaf (parameter).

        Once we reach a leaf, we calculate its gradient by applying
        the gradient function for each operation that we passed along the
        path
        """

        queue = [self]
        labels = {}
        edge_labels = {}

        G = nx.DiGraph()
        G.add_node(str(self))
        labels[str(self)] = self.name or ""

        logger.info("\n\n")
        while queue:
            logger.info(f"stack: {queue}")
            current_node = queue.pop()

            # At intermediate node
            if (
                current_node.back_fn is not None
                and current_node.args is not None  # noqa: E501
            ):
                # Get gradient functions for the operation
                grad_fn = current_node.grad_fn

                logger.info(
                    f"{current_node.args=} {current_node.back_fn=} { current_node=}"
                )
                # Update gradient functions for each parent node
                for arg, op in zip(current_node.args, current_node.back_fn):
                    if str(arg) not in G.nodes:
                        G.add_node(str(arg))
                        labels[str(arg)] = arg.name or ""
                    G.add_edge(str(arg), str(current_node))

                    try:
                        edge_labels[(str(arg), str(current_node))] = op.__name__
                    except AttributeError:
                        edge_labels[(str(arg), str(current_node))] = op.func.__name__

                    arg.grad_fn = grad_fn + [op]

                    # Add each arg to the stack for further processing
                    queue = [arg] + queue

            # At leaf node
            elif current_node.requires_grad:
                grad = tensor(np.ones_like(self))
                logger.warning(current_node.grad_fn)
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

        fig, ax = plt.subplots(figsize=(10, 10))
        pos = graphviz_layout(G, prog="dot")
        nx.draw(G, labels=labels, pos=pos, ax=ax)
        nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels, ax=ax)
        plt.show()

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
        return f"{super().__repr__()[:-1]}, {self.name=})"


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


# -----------------utility functions--------


def tensor(
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


# ----------binary functions----------------


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
    # scalar / tensor or tensor / scalar
    if len(x.shape) == 0 or len(y.shape) == 0:
        minus_one = tensor(-1.0, requires_grad=False, name="minus_one")
        return mul(x, pow(y, minus_one))

    # tensor / tensor
    elif len(x.shape) == len(y.shape):
        one = tensor(1.0, requires_grad=False, name="one")
        indices = ascii_lowercase[: len(x.shape)]
        einsum(x, one / y, subscripts=f"{indices},{indices}->{indices}")
    else:
        raise ValueError(
            "Division between two tensors is of different shapes is not supported."
        )


# -----------elementwise unary functions----------------


@to_tensor
def pow(x: Tensor, y: Tensor) -> Tensor:
    """
    Raise every element of a tensor to the power of another tensor
    """
    global grad
    result = tensor(np.power(x, y))
    if not grad:
        return result

    result.back_fn = (
        partial(mul, y=tensor(np.power(x, y - 1)) * y),
        partial(mul, y=result * log(x)),
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
    result.back_fn = (partial(mul, y=result),)
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

    result.back_fn = (partial(div, y=x),)
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


# ------------------reduce functions----------------


def reduce(x: Tensor, method: Op, subscripts: str) -> Tensor:
    """
    Reduce a tensor along some dimensions by applying a reduction function
    to those indices. A reduction function generates a binary tensor which
    is multiplied by the input and reduces according to the subscripts
    """
    indices, output = _parse_subscripts(subscripts)
    assert (
        len(indices) == 1
    ), f"Can only reduce a single tensor at a time. Indices suggeststed: {len(indices)} tensors: {indices}"

    [idx] = indices
    axis = tuple(i for i, char in enumerate(idx) if char not in output)
    assert axis, "Tensor must be reduced along at least one axis"
    binary_tensor = method(x, axis)
    binary_tensor.requires_grad = False
    binary_tensor.name = "binary"

    subscripts = f"{idx},{idx}->{output}"

    return einsum(x, binary_tensor, subscripts=subscripts)


# --------------------indicator functions---------------------
def bmax(x: Tensor, axis: Union[int, Tuple[int]]) -> Tensor:
    """
    Return a binary tensor where each element is 1 if the corresponding element
    is that largest along an axis that is being reduced along
    """
    return (x == np.max(x, axis=axis, keepdims=True)).astype(int)


def bmin(x: Tensor, axis: Union[int, Tuple[int]]) -> Tensor:
    """
    Return a binary tensor where each element is 1 if the corresponding element
    is that smallest along an axis that is being reduced along
    """
    return (x == np.min(x, axis=axis, keepdims=True)).astype(int)


@to_tensor
def max(x: Tensor) -> Tensor:
    """
    Find the largest element of a tensor
    """
    indices = ascii_lowercase[: len(x.shape)]
    return reduce(x, bmax, f"{indices}->")


@to_tensor
def min(x: Tensor) -> Tensor:
    """
    Find the smallest element of a tensor
    """
    indices = ascii_lowercase[: len(x.shape)]
    return reduce(x, bmin, f"{indices}->")


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
    do deep learningbinary_tensor,

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

    indices, output = _parse_subscripts(subscripts)

    # Every single input tensor is equivalent to a dual input where
    # the second input is a 1 tensor with the indices that dont appear in the
    # output
    if len(tensors) == 1:
        one_indices = ""
        one_shape = []
        for size, idx in zip(tensors[0].shape, indices[0]):
            if idx not in output:
                one_indices += idx
                one_shape.append(size)

        indices = [indices[0], one_indices]
        ones = tensor(np.ones(one_shape), requires_grad=False, name="ones")
        tensors = (tensors[0], ones)

    if not tensors or len(tensors) > 2:
        raise NotImplementedError(
            f"Einsum is only implemented for 1 or 2 inputs. Found {len(tensors)}"
        )

    indices = ",".join(indices)
    subscripts = f"{indices}->{output}"

    result = tensor(np.einsum(subscripts, *tensors))

    if not grad:
        return result

    back_fns = []
    indices, output = _parse_subscripts(subscripts)

    # The rule for differentiating einsum wrt one of its arguments is:
    # Swap the indices of the output and the argument being differentiated
    # in the subscripts. Then pass the current gradient as an argument in
    # place of the argument being differentiated.
    # for example:
    # >>> a = np.arange(12).reshape(3, 4)
    # >>> b = np.arange(12).reshape(4, 3)
    # >>> c = np.einsum('ij,jk->ik', a, b)
    #
    # >>> c.backward()
    # >>> assert a.grad == np.einsum('ik,jk->ij', np.ones_like(c), b)
    # Notice that the indices have swapped and a has been replaced with
    # the gradient of c (the current gradient)
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
    indices = ascii_letters[: len(x.shape)]
    largest_x = reduce(x, bmax, subscripts=f"{indices}->{indices[:-1]}")
    ones = tensor(np.ones_like(x), requires_grad=False, name="ones")

    largest_x = einsum(
        largest_x, ones, subscripts=f"{indices[:-1]},{indices}->{indices}"
    )
    normalised_x = x - largest_x

    normalised_x = exp(normalised_x)
    assert len(normalised_x.shape) < len(
        ascii_letters
    ), f"Cannot perform softmax on tensors with more than {len(ascii_letters)} dimensions"

    if len(x.shape) == 1:
        one = tensor(1, name="one", requires_grad=False)
        denom = one / einsum(normalised_x, subscripts="i->")
        return einsum(denom, normalised_x, subscripts=",i->i")

    indices = ascii_letters[: len(x.shape) - 1]
    final_letter = ascii_letters[-1]
    denom = 1 / einsum(
        normalised_x, subscripts=f"{final_letter}{indices}->{final_letter}"
    )
    return einsum(
        denom,
        normalised_x,
        subscripts=f"{final_letter},{final_letter}{indices}->{final_letter}{indices}",
    )


@to_tensor
def sigmoid(x: Tensor) -> Tensor:
    """
    Compute the sigmoid of a tensor
    """
    return tensor(1) / (exp(-x) + 1)
