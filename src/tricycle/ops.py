from typing import Sequence

from tricycle.einsum import Einsum, Subscript
from tricycle.tensor import Tensor, to_tensor
from tricycle.unary import usqrt


def repeat(tensor: Tensor, repeats: int):
    """
    Repeat a tensor along its final axis
    This is done my multiplying with a ones tensor the same shape as the
    desired output
    """
    xp = tensor.xp
    subscript = Subscript("...,...a->...a")
    new_shape = tensor.shape + (repeats,)
    ones = to_tensor(
        xp.ones(new_shape), is_vector=tensor.is_vector, requires_grad=False
    )

    return Einsum(subscript)(tensor, ones)


def split(tensor: Tensor, n_splits: int, axis: int = 0) -> Sequence[Tensor]:
    """
    Split a tensor along its first axis into n_splits partitions
    """
    xp = tensor.xp
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

        result = to_tensor(
            tensor[tuple(idx)], requires_grad=tensor.requires_grad
        )

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
                xp.zeros(tensor.shape), is_vector=result.is_vector
            )
            result_grad[tuple(idx)] = grad._data
            return result_grad

        result.back_fns = (undo_split,)
        result.args = (tensor,)
        result.is_vector = tensor.is_vector
        results.append(result)
    return results


def reshape(tensor: Tensor, shape: Sequence[int]):
    xp = tensor.xp
    if tensor.is_vector:
        shape = [tensor.shape[0]] + list(shape)

    result = to_tensor(xp.reshape(tensor._data, shape))
    result.is_vector = tensor.is_vector
    result.args = (tensor,)

    def undo_reshape(grad):
        new_shape = tensor.shape[1:] if tensor.is_vector else tensor.shape
        return reshape(grad, new_shape)

    result.back_fns = (undo_reshape,)

    return result


def mean(tensor: Tensor) -> Tensor:
    """
    Find the mean of a tensor
    """
    return tensor.e("...a->...") / tensor.shape[-1]


def variance(tensor: Tensor) -> Tensor:
    average = mean(tensor).repeat(tensor.shape[-1])
    square_deviation = (tensor - average) ** 2
    return mean(square_deviation)


def standard_deviation(tensor: Tensor) -> Tensor:
    """
    Find the standard deviation of a tensor
    """
    return usqrt(variance(tensor))


def normalise(tensor: Tensor, eps: float = 1e-8) -> Tensor:
    """
    Standard normalise a tensor. Optionally add a small constant to the
    divisor to avoid division by zero.
    """
    x_mean = mean(tensor).repeat(tensor.shape[-1])
    x_standard_deviation = standard_deviation(tensor).repeat(tensor.shape[-1])
    if eps:
        x_standard_deviation += eps
    return (tensor - x_mean) / x_standard_deviation
