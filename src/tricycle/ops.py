from typing import Sequence

import numpy as np

from tricycle.einsum import Einsum, Subscript
from tricycle.reduce import rmax
from tricycle.tensor import Tensor, to_tensor


def repeat(tensor: Tensor, repeats: int):
    """
    Repeat a tensor along its final axis
    This is done my multiplying with a ones tensor the same shape as the
    desired output
    """
    subscript = Subscript("...,...a->...a")
    new_shape = tensor.shape + (repeats,)
    ones = to_tensor(
        np.ones(new_shape), is_vector=tensor.is_vector, requires_grad=False
    )

    return Einsum(subscript)(tensor, ones)


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
    from tricycle.binary import bdiv
    from tricycle.unary import uexp

    # normalise
    largest_element = rmax(tensor, "...a->...").repeat(tensor.shape[-1])
    tensor = tensor - largest_element

    numerator = uexp(tensor)
    denominator = numerator.e("...a->...").repeat(tensor.shape[-1])
    return bdiv(numerator, denominator)


def arange(*args, **kwargs):
    return to_tensor(np.arange(*args, **kwargs))


def split(tensor: Tensor, n_splits: int, axis: int = 0) -> Sequence[Tensor]:
    """
    Split a tensor along its first axis int n_splits partitions
    """
    if axis < 0:
        axis += tensor.ndim
    if tensor.is_vector:
        axis += 1

    length = tensor.shape[axis]
    if length % n_splits:
        raise ValueError(
            f"Length must be divisible by n_splits. Found {length} and {n_splits}"
        )
    split_size = length // n_splits

    results = []
    for split_idx in range(n_splits):
        idx = []
        for i in range(tensor.ndim):
            if i == axis:
                axis_idx = slice(
                    split_idx * split_size, (split_idx + 1) * split_size
                )
            else:
                axis_idx = slice(None)
            idx.append(axis_idx)

        result = tensor[*idx]

        def undo_split(grad, idx=idx):
            """
            The backwards operation for a split operation.
            Produces a tensor of zeros the same shape as the input
            except in the section that was split

            e.g
            >>> result = split([1,2,3,4], 2)
            >>> result
            [tensor([1, 2]), tensor([3, 4])]
            # set an arbitrary derivative for first split
            >>> result[0].grad = to_tensor([1,1])
            >>> undo_split(result[0].grad)
            [1, 1, 0, 0]
            """
            result_grad = to_tensor(
                np.zeros_like(tensor), is_vector=result.is_vector
            )
            result_grad[*idx] = grad
            return result_grad

        result.back_fn = (undo_split,)
        result.args = (tensor,)
        result.is_vector = tensor.is_vector
        results.append(result)
    return results


def reshape(tensor: Tensor, shape: Sequence[int]):
    if tensor.is_vector:
        shape = [tensor.shape[0]] + list(shape)

    result = to_tensor(np.reshape(tensor, shape))
    result.is_vector = tensor.is_vector
    result.args = (tensor,)
    result.back_fn = (lambda grad: reshape(grad, tensor.shape),)

    return result
