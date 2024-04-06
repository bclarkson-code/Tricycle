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

    assert len(subscript.inputs) == 1, (
        "Can only reduce a single tensor at a time."
        f"Indices suggeststed: {len(subscript.inputs)}"
        f" tensors: {subscript.inputs}"
    )
    [idx] = subscript.inputs

    # figure out which indices we need to reduce along, handling ...
    i = 0
    reduce_along_axes = []
    for char in idx:
        if char == "...":
            n_hidden_indices = tensor.ndim - len(idx) + 1

            if char not in subscript.output:
                reduce_along_axes.extend(list(range(i, i + n_hidden_indices)))

            i += n_hidden_indices
            continue

        if char not in subscript.output:
            reduce_along_axes.append(i)
        i += 1

    if not reduce_along_axes:
        return tensor

    indicator = tensor._data == np.max(
        tensor._data, axis=tuple(reduce_along_axes), keepdims=True
    )
    indicator = to_tensor(
        indicator, requires_grad=False, is_vector=tensor.is_vector
    )
    indicator._data = indicator._data.astype(np.int8)

    new_subscript = Subscript.from_split([idx, idx], subscript.output)

    # we're allowed to replace inf with a large number here because
    # we're multiplying with a binary vector. In other circumstances
    # this would might break things
    result = Einsum(new_subscript)(tensor, indicator, replace_inf=True)
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

    indicator = tensor._data == np.min(
        tensor._data, axis=tuple(reduce_along_axes), keepdims=True
    )
    indicator = to_tensor(
        indicator, requires_grad=False, is_vector=tensor.is_vector
    )
    indicator._data = indicator._data.astype(np.int8)

    new_subscript = Subscript.from_split([idx, idx], subscript.output)

    result = Einsum(new_subscript)(tensor, indicator)
    result.name = f"min({new_subscript})"

    return result
