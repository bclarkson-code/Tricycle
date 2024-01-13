from functools import partial
from typing import Optional

import numpy as np

from tricycle_v2.tensor import Tensor


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


def apply(tensor, op, *args, **kwargs):
    """
    Apply a unary operation elementwise to a tensor
    """
    return op(tensor, *args, **kwargs)


def apply_binary(tensor_1, tensor_2, op, *args, **kwargs):
    """
    Apply a binary operation elementwise to a tensor
    """
    return op(tensor_1, tensor_2, *args, **kwargs)


def reduce(tensor, indices, op, *args, **kwargs):
    """
    Reduce a tensor along some dimensions by applying a reduction function
    to those indices. A reduction function
    """
    return op(tensor, indices, *args, **kwargs)


def einsum(subscripts, tensor_1, tensor_2):
    """
    Use einstein notation to combine two tensors
    """
    indices, output = _parse_subscripts(subscripts)
    if len(indices) != 2:
        raise NotImplementedError("Can only perform einsum on two tensors")

    result = to_tensor(np.einsum(subscripts, tensor_1, tensor_2))

    # The rule for differentiating einsum is to swap the indices
    # of the output and input being differentiated
    left_subscript = f"{output},{indices[1]}->{indices[0]}"
    right_subscript = f"{indices[0]},{output}->{indices[1]}"
    result.args = (tensor_1, tensor_2)

    left_back_fn = partial(einsum, left_subscript, tensor_2=tensor_2)
    right_back_fn = partial(einsum, right_subscript, tensor_1)
    result.back_fn = (left_back_fn, right_back_fn)

    return result


def _parse_subscripts(subscripts: str) -> tuple[list[str], str]:
    """
    Parse a subscripts string into a list of indices and a result
    """
    indices, result = subscripts.split("->")
    indices = indices.split(",")
    return indices, result


def nothing(tensor):
    """
    Return a tensor
    """
    return tensor
