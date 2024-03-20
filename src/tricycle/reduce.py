import numpy as np

from tricycle.einsum import Einsum, Subscript
from tricycle.tensor import Tensor, to_tensor




def rmax(tensor: Tensor, subscript: str | Subscript):
    """
    Generate an indicator tensor that, when einsummed with the tensor, results
    in a tensor that is equal to the result of max applied along the indices
    that dont appear in the output of the subscript
    """
    if isinstance(subscript, str):
        subscript = Subscript(subscript)

    assert (
        len(subscript.inputs) == 1
    ), f"Can only reduce a single tensor at a time. Indices suggeststed: {len(subscript.inputs)} tensors: {subscript.inputs}"
    [idx] = subscript.inputs

    reduce_along_axes = [
        i for i, char in enumerate(idx) if char not in subscript.output
    ]

    if not reduce_along_axes:
        return tensor

    indicator = (
        tensor == np.max(tensor, axis=tuple(reduce_along_axes), keepdims=True)
    ).astype(int)
    indicator = to_tensor(indicator, requires_grad=False, is_vector=tensor.is_vector)

    new_subscript = f"{idx},{idx}->{subscript.output}"

    result = Einsum(new_subscript)(tensor, indicator)
    result.name = f"min({new_subscript})"

    return result


def rmin(tensor: Tensor, subscript: Subscript | str):
    """
    Generate an indicator tensor that, when einsummed with the tensor, results
    in a tensor that is equal to the result of min applied along the indices
    that dont appear in the output of the subscript
    """
    if isinstance(subscript, str):
        subscript = Subscript(subscript)

    assert (
        len(subscript.inputs) == 1
    ), f"Can only reduce a single tensor at a time. Indices suggeststed: {len(subscript.inputs)} tensors: {subscript.inputs}"
    [idx] = subscript.inputs

    reduce_along_axes = [
        i for i, char in enumerate(idx) if char not in subscript.output
    ]

    if not reduce_along_axes:
        return tensor

    indicator = (
        tensor == np.min(tensor, axis=tuple(reduce_along_axes), keepdims=True)
    ).astype(int)
    indicator = to_tensor(indicator, requires_grad=False, is_vector=tensor.is_vector)

    new_subscript = Subscript.from_split([idx, idx], subscript.output)

    result = Einsum(new_subscript)(tensor, indicator)
    result.name = f"min({new_subscript})"

    return result
