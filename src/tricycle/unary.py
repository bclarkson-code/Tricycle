from copy import copy
from functools import partial
from string import ascii_letters

import numpy as np

from tricycle.ops import Einsum, nothing
from tricycle.tensor import Tensor, to_tensor

grad = False


def uadd(tensor: Tensor, constant: float) -> Tensor:
    """
    Add a constant, elementwise, to a tensor. The constant is not
    differentiable.
    """
    assert isinstance(tensor, Tensor)
    assert np.isscalar(constant)

    result = to_tensor(np.add(tensor, constant))
    result.args = (tensor,)
    result.back_fn = (nothing,)
    result.name = f"+ {constant}"
    result.is_vector = tensor.is_vector
    return result


def umul(tensor: Tensor, constant: float) -> Tensor:
    """
    Multiply a constant, elementwise, to a tensor. The constant is not
    differentiable.
    """
    assert isinstance(tensor, Tensor)
    assert np.isscalar(constant)

    constant_tensor = to_tensor(
        np.full_like(tensor, constant, dtype=float), requires_grad=False
    )
    constant_tensor.is_vector = tensor.is_vector
    return Einsum("a,a->a")(tensor, constant_tensor)


def usub(tensor: Tensor, constant: float) -> Tensor:
    """
    Subtract a constant, elementwise, from a tensor. The constant is not
    differentiable.
    """
    return uadd(tensor, -constant)


def upow(tensor: Tensor, constant: float) -> Tensor:
    """
    Raise a tensor to a constant, elementwise. The constant is not
    differentiable.
    """
    from tricycle.binary import bmul

    assert isinstance(tensor, Tensor)
    assert np.isscalar(constant)

    result = to_tensor(np.power(tensor, constant))
    result.args = (tensor,)
    coef = to_tensor(np.power(tensor, constant - 1))
    coef.is_vector = tensor.is_vector
    result.back_fn = (partial(bmul, umul(coef, constant)),)
    result.name = f"^ {constant}"
    result.is_vector = tensor.is_vector

    return result


def udiv(constant: float, tensor: Tensor) -> Tensor:
    """
    Divide a constant by a tensor, elementwise. The constant is not
    differentiable.
    """
    return umul(upow(tensor, -1.0), constant)


def umax(tensor: Tensor, constant: float) -> Tensor:
    """
    Return the max of the tensor and the constant,
    elementwise. The constant is not differentiable.
    """
    assert isinstance(tensor, Tensor)
    assert np.isscalar(constant)

    result = to_tensor(np.maximum(tensor, constant))

    from tricycle.binary import bmul

    result.args = (tensor,)
    is_bigger = to_tensor((tensor > constant).astype(float))
    result.back_fn = (partial(bmul, is_bigger),)
    result.name = f"> {constant}"
    result.is_vector = tensor.is_vector

    return result


def umin(tensor: Tensor, constant: float) -> Tensor:
    """
    Return the max of the tensor and the constant,
    elementwise. The constant is not differentiable.
    """
    assert isinstance(tensor, Tensor)
    assert np.isscalar(constant)

    result = to_tensor(np.minimum(tensor, constant))

    from tricycle.binary import bmul

    result.args = (tensor,)
    is_smaller = to_tensor((tensor < constant).astype(float))
    result.back_fn = (partial(bmul, is_smaller),)
    result.name = f"< {constant}"
    result.is_vector = tensor.is_vector
    return result


def uexp(tensor: Tensor) -> Tensor:
    """
    Raise every element of a tensor to the power of e
    """
    result = to_tensor(np.exp(tensor))

    from tricycle.binary import bmul

    result.args = (tensor,)
    result.back_fn = (partial(bmul, copy(result)),)
    result.name = "exp"
    result.is_vector = tensor.is_vector
    return result


def ulog(tensor: Tensor) -> Tensor:
    """
    Raise every element of a tensor to the power of e
    """
    result = to_tensor(np.log(tensor))

    from tricycle.binary import bmul

    result.args = (tensor,)
    result.back_fn = (
        partial(
            bmul,
            udiv(1, tensor),
        ),
    )
    result.name = "log"
    result.is_vector = tensor.is_vector
    return result


def usin(tensor: Tensor) -> Tensor:
    """
    Raise every element of a tensor to the power of e
    """
    result = to_tensor(np.sin(tensor))

    from tricycle.binary import bmul

    result.args = (tensor,)
    coef = to_tensor(np.cos(tensor))
    result.back_fn = (partial(bmul, coef),)
    result.name = "sin"
    result.is_vector = tensor.is_vector
    return result


def ucos(tensor: Tensor) -> Tensor:
    """
    Raise every element of a tensor to the power of e
    """
    result = to_tensor(np.cos(tensor))

    from tricycle.binary import bmul

    result.args = (tensor,)
    coef = to_tensor(-np.sin(tensor))
    result.back_fn = (partial(bmul, coef),)
    result.name = "cos"
    result.is_vector = tensor.is_vector
    return result
