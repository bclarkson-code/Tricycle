from functools import partial

from tricycle.ops import Einsum
from tricycle.tensor import Tensor, nothing, select_backend, to_tensor
from tricycle.unary import udiv, umul


def _shapes_match(tensor_1: Tensor, tensor_2: Tensor) -> bool:
    # sourcery skip: assign-if-exp, merge-duplicate-blocks, remove-redundant-if
    if tensor_1.is_vector and tensor_2.is_vector:
        shape_1 = tensor_1.shape
        shape_2 = tensor_2.shape
    elif tensor_1.is_vector:
        shape_1 = tensor_1.shape[1:]
        shape_2 = tensor_2.shape
    elif tensor_2.is_vector:
        shape_1 = tensor_1.shape
        shape_2 = tensor_2.shape[1:]
    else:
        shape_1 = tensor_1.shape
        shape_2 = tensor_2.shape

    if shape_1 != shape_2:
        raise ValueError(
            f"Shapes {shape_1} and {shape_2} do not match: {tensor_1._data.shape}, {tensor_2._data.shape}"
        )
    return shape_1 == shape_2


def badd(tensor_1: Tensor, tensor_2: Tensor) -> Tensor:
    """
    Add the elements of two tensors together, elementwise

    The two tensors must have the same shape
    """
    xp = select_backend(tensor_1._data, tensor_2._data)
    assert _shapes_match(tensor_1, tensor_2)

    result = to_tensor(xp.add(tensor_1._data, tensor_2._data))

    result.args = (tensor_1, tensor_2)
    result.back_fns = (nothing, nothing)
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

    result = Einsum("...,...->...")(tensor_1, tensor_2)
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
    xp = select_backend(tensor_1._data, tensor_2._data)
    assert _shapes_match(tensor_1, tensor_2)

    result = to_tensor(xp.maximum(tensor_1._data, tensor_2._data))

    indicator_1 = tensor_1 > tensor_2
    indicator_1.is_vector = tensor_1.is_vector

    indicator_2 = tensor_1 <= tensor_2
    indicator_2.is_vector = tensor_2.is_vector

    result.args = (tensor_1, tensor_2)
    result.back_fns = (partial(bmul, indicator_1), partial(bmul, indicator_2))
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
    xp = select_backend(tensor_1._data, tensor_2._data)
    assert _shapes_match(tensor_1, tensor_2)

    result = to_tensor(xp.minimum(tensor_1._data, tensor_2._data))

    indicator_1 = tensor_1 < tensor_2
    indicator_1.is_vector = tensor_1.is_vector

    indicator_2 = tensor_1 >= tensor_2
    indicator_2.is_vector = tensor_2.is_vector

    result.args = (tensor_1, tensor_2)
    result.back_fns = (partial(bmul, indicator_1), partial(bmul, indicator_2))
    result.name = "bmin"
    result.is_vector = tensor_1.is_vector or tensor_2.is_vector

    return result


def bmask(tensor: Tensor, mask: Tensor) -> Tensor:
    """
    Apply a binary mask to a numpy array, setting values to 0 where
    the mask is True
    """
    xp = select_backend(tensor._data, mask._data)
    assert _shapes_match(tensor, mask)
    assert (
        mask.requires_grad == False
    ), "Cannot compute gradient of a binary mask"

    result = to_tensor(xp.where(mask._data, tensor._data, 0))

    result.args = (tensor,)
    result.back_fns = (lambda grad: bmask(grad, mask),)
    result.name = "bmask"
    result.is_vector = tensor.is_vector
    return result
