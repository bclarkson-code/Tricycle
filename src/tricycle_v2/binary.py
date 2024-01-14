from functools import partial
from string import ascii_letters

import numpy as np

from tricycle_v2.ops import einsum, nothing
from tricycle_v2.tensor import Tensor, to_tensor
from tricycle_v2.unary import udiv, umul


def badd(tensor_1: Tensor, tensor_2: Tensor) -> Tensor:
    """
    Add the elements of two tensors together, elementwise

    The two tensors must have the same shape
    """
    assert tensor_1.shape == tensor_2.shape

    result = to_tensor(np.add(tensor_1, tensor_2))

    result.args = (tensor_1, tensor_2)
    result.back_fn = (nothing, nothing)

    return result


def bsub(tensor_1: Tensor, tensor_2: Tensor) -> Tensor:
    """
    Subtract the elements of two tensors together, elementwise

    The two tensors must have the same shape
    """
    assert tensor_1.shape == tensor_2.shape

    tensor_2_neg = umul(tensor_2, -1)
    return badd(tensor_1, tensor_2_neg)


def bmul(tensor_1: Tensor, tensor_2: Tensor) -> Tensor:
    """
    Multiply the elements of two tensors together, elementwise

    The two tensors must have the same shape
    """
    assert tensor_1.shape == tensor_2.shape

    indices = ascii_letters[: len(tensor_1.shape)]
    subscripts = f"{indices},{indices}->{indices}"

    return einsum(subscripts, tensor_1, tensor_2)


def bdiv(tensor_1: Tensor, tensor_2: Tensor) -> Tensor:
    """
    Divide the elements of two tensors together, elementwise

    The two tensors must have the same shape
    """
    return bmul(tensor_1, udiv(1, tensor_2))


def bmax(tensor_1: Tensor, tensor_2: Tensor) -> Tensor:
    """
    Compare two tensors elementwise, returning the maximum
    of each pair of elements

    The two tensors must have the same shape
    if elements are equal, return the first
    """
    assert tensor_1.shape == tensor_2.shape

    result = to_tensor(np.maximum(tensor_1, tensor_2))

    indicator_1 = to_tensor((tensor_1 > tensor_2).astype(float))
    indicator_2 = to_tensor((tensor_1 <= tensor_2).astype(float))
    result.args = (tensor_1, tensor_2)
    result.back_fn = (partial(bmul, indicator_1), partial(bmul, indicator_2))

    return result


def bmin(tensor_1: Tensor, tensor_2: Tensor) -> Tensor:
    """
    Compare two tensors elementwise, returning the minimum
    of each pair of elements

    The two tensors must have the same shape
    if elements are equal, return the first
    """
    assert tensor_1.shape == tensor_2.shape

    result = to_tensor(np.minimum(tensor_1, tensor_2))

    indicator_1 = to_tensor((tensor_1 < tensor_2).astype(float))
    indicator_2 = to_tensor((tensor_1 >= tensor_2).astype(float))
    result.args = (tensor_1, tensor_2)
    result.back_fn = (partial(bmul, indicator_1), partial(bmul, indicator_2))

    return result
