from functools import partial

import numpy as np

from tricycle.ops import Einsum, nothing
from tricycle.tensor import Tensor, to_tensor
from tricycle.unary import udiv, umul


def _shapes_match(tensor_1: Tensor, tensor_2: Tensor) -> bool:
    # sourcery skip: assign-if-exp
    if tensor_1.is_vector:
        shape_1 = tensor_1.shape[1:]
    else:
        shape_1 = tensor_1.shape

    if tensor_2.is_vector:
        shape_2 = tensor_2.shape[1:]
    else:
        shape_2 = tensor_2.shape

    if shape_1 != shape_2:
        breakpoint()
    return shape_1 == shape_2


def badd(tensor_1: Tensor, tensor_2: Tensor) -> Tensor:
    """
    Add the elements of two tensors together, elementwise

    The two tensors must have the same shape
    """
    assert _shapes_match(tensor_1, tensor_2)

    result = to_tensor(np.add(tensor_1, tensor_2))

    result.args = (tensor_1, tensor_2)
    result.back_fn = (nothing, nothing)
    result.name = "badd"

    if tensor_1.is_vector or tensor_2.is_vector:
        result.is_vector = True

    return result


def bsub(tensor_1: Tensor, tensor_2: Tensor) -> Tensor:
    """
    Subtract the elements of two tensors together, elementwise

    The two tensors must have the same shape
    """
    assert _shapes_match(tensor_1, tensor_2)

    return badd(tensor_1, umul(tensor_2, -1))


def bmul(tensor_1: Tensor, tensor_2: Tensor) -> Tensor:
    """
    Multiply the elements of two tensors together, elementwise

    The two tensors must have the same shape
    """
    assert _shapes_match(tensor_1, tensor_2)

    result = Einsum("a,a->a")(tensor_1, tensor_2)
    result.name = "bmul"
    return result


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
    assert _shapes_match(tensor_1, tensor_2)

    result = to_tensor(np.maximum(tensor_1, tensor_2))

    indicator_1 = to_tensor(
        (tensor_1 > tensor_2).astype(float), is_vector=tensor_1.is_vector
    )
    indicator_2 = to_tensor(
        (tensor_1 <= tensor_2).astype(float), is_vector=tensor_2.is_vector
    )
    result.args = (tensor_1, tensor_2)
    result.back_fn = (partial(bmul, indicator_1), partial(bmul, indicator_2))
    result.name = "bmax"
    result.is_vector = tensor_1.is_vector or tensor_2.is_vector
    return result


def bmin(tensor_1: Tensor, tensor_2: Tensor) -> Tensor:
    """
    Compare two tensors elementwise, returning the minimum
    of each pair of elements

    The two tensors must have the same shape
    if elements are equal, return the first
    """
    assert _shapes_match(tensor_1, tensor_2)

    result = to_tensor(np.minimum(tensor_1, tensor_2))

    indicator_1 = to_tensor(
        (tensor_1 < tensor_2).astype(float), is_vector=tensor_1.is_vector
    )
    indicator_2 = to_tensor(
        (tensor_1 >= tensor_2).astype(float), is_vector=tensor_2.is_vector
    )
    result.args = (tensor_1, tensor_2)
    result.back_fn = (partial(bmul, indicator_1), partial(bmul, indicator_2))
    result.name = "bmin"
    result.is_vector = tensor_1.is_vector or tensor_2.is_vector

    return result
