import numpy as np

from tricycle_v2.ops import _parse_subscripts, einsum, to_tensor
from tricycle_v2.tensor import Tensor


def radd(tensor: Tensor, subscript: str):
    """
    Generate an indicator tensor that, when einsummed with the tensor, results
    in a tensor that is equal to the result of summing along the indices
    that dont appear in the output of the subscript
    """
    indices, output = _parse_subscripts(subscript)
    assert (
        len(indices) == 1
    ), f"Can only reduce a single tensor at a time. Indices suggeststed: {len(indices)} tensors: {indices}"
    [idx] = indices

    indicator_indices = ""
    reduce_along_axes = []
    for i, char in enumerate(idx):
        if char not in output:
            indicator_indices += char
            reduce_along_axes.append(i)

    if not reduce_along_axes:
        return tensor

    indicator_shape = [tensor.shape[i] for i in reduce_along_axes]
    indicator = to_tensor(np.ones(indicator_shape, dtype=np.bool_), requires_grad=False)

    new_subscript = f"{idx},{indicator_indices}->{output}"
    return einsum(new_subscript, tensor, indicator)


def rmax(tensor: Tensor, subscript: str):
    """
    Generate an indicator tensor that, when einsummed with the tensor, results
    in a tensor that is equal to the result of max applied along the indices
    that dont appear in the output of the subscript
    """
    indices, output = _parse_subscripts(subscript)
    assert (
        len(indices) == 1
    ), f"Can only reduce a single tensor at a time. Indices suggeststed: {len(indices)} tensors: {indices}"
    [idx] = indices

    reduce_along_axes = [i for i, char in enumerate(idx) if char not in output]

    if not reduce_along_axes:
        return tensor

    indicator = (
        tensor == np.max(tensor, axis=tuple(reduce_along_axes), keepdims=True)
    ).astype(int)

    new_subscript = f"{idx},{idx}->{output}"

    return einsum(new_subscript, tensor, indicator)


def rmin(tensor: Tensor, subscript: str):
    """
    Generate an indicator tensor that, when einsummed with the tensor, results
    in a tensor that is equal to the result of min applied along the indices
    that dont appear in the output of the subscript
    """
    indices, output = _parse_subscripts(subscript)
    assert (
        len(indices) == 1
    ), f"Can only reduce a single tensor at a time. Indices suggeststed: {len(indices)} tensors: {indices}"
    [idx] = indices

    reduce_along_axes = [i for i, char in enumerate(idx) if char not in output]

    if not reduce_along_axes:
        return tensor

    indicator = (
        tensor == np.min(tensor, axis=tuple(reduce_along_axes), keepdims=True)
    ).astype(int)

    new_subscript = f"{idx},{idx}->{output}"

    return einsum(new_subscript, tensor, indicator)
