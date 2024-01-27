from functools import partial
from string import ascii_lowercase

import numpy as np

from tricycle.tensor import to_tensor


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
    result.name = f"einsum {subscripts}"

    return result


def repeat(subscripts, tensor, out_shape):
    """
    Repeat a tensor along some indices, according to the subscript.
    Note: This is mathematically equivalent to einsumming the tensor
    with a one tensor
    """
    indices, output = _parse_subscripts(subscripts)
    assert len(indices) == 1
    [index] = indices

    assert len(output) == len(out_shape), "Output shape does not match subscripts"

    one_indices = ""
    one_shape = []
    for size, out_idx in zip(out_shape, output):
        if out_idx not in index:
            one_indices += out_idx
            one_shape.append(size)

    ones = to_tensor(np.ones(one_shape), requires_grad=False)
    new_subscript = f"{one_indices},{index}->{output}"
    return einsum(new_subscript, ones, tensor)


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


def softmax(tensor):
    """
    Apply softmax. The softmax is only applied to the final
    dimension of the tensor
    Note: the tensor is normalised for numeric stability
    """
    from tricycle.reduce import radd
    from tricycle.binary import bdiv
    from tricycle.unary import uexp

    indices = ascii_lowercase[: len(tensor.shape)]
    reduce_subscript = f"{indices}->{indices[:-1]}"

    expand_subscript = f"{indices[:-1]}->{indices}"
    # largest = repeat(expand_subscript, largest, tensor.shape)
    normalised = tensor  #  - largest
    exponentiated = uexp(normalised)

    denom = radd(exponentiated, reduce_subscript)
    denom = repeat(expand_subscript, denom, tensor.shape)
    return bdiv(exponentiated, denom)
