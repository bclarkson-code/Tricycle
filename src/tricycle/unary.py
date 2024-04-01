from functools import partial

import numpy as np
from scipy.special import erf

from tricycle.einsum import Einsum
from tricycle.tensor import Tensor, nothing, to_tensor

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
    result.back_fns = (nothing,)
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
        np.full_like(tensor, constant, dtype=float),
        requires_grad=False,
        is_vector=tensor.is_vector,
    )

    return Einsum("...,...->...")(tensor, constant_tensor)


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
    coef = to_tensor(
        np.power(tensor, constant - 1), is_vector=tensor.is_vector
    )
    result.back_fns = (partial(bmul, umul(coef, constant)),)
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
    is_bigger = to_tensor(
        (tensor > constant).astype(float), is_vector=tensor.is_vector
    )
    result.back_fns = (partial(bmul, is_bigger),)
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
    is_smaller = to_tensor(
        (tensor < constant).astype(float), is_vector=tensor.is_vector
    )
    result.back_fns = (partial(bmul, is_smaller),)
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
    result.name = "exp"
    result.is_vector = tensor.is_vector
    coef = to_tensor(result, is_vector=tensor.is_vector)
    result.back_fns = (partial(bmul, coef),)
    return result


def ulog(tensor: Tensor) -> Tensor:
    """
    Raise every element of a tensor to the power of e
    """
    result = to_tensor(np.log(tensor))

    from tricycle.binary import bmul

    result.args = (tensor,)
    result.back_fns = (
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
    coef = to_tensor(np.cos(tensor), is_vector=tensor.is_vector)
    result.back_fns = (partial(bmul, coef),)
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
    coef = to_tensor(-np.sin(tensor), is_vector=tensor.is_vector)
    result.back_fns = (partial(bmul, coef),)
    result.name = "cos"
    result.is_vector = tensor.is_vector
    return result


def usqrt(tensor: Tensor):
    """
    Apply the square root function
    """
    return upow(tensor, 0.5)


def uerf(tensor: Tensor) -> Tensor:
    """
    Calculate the error function of every element of a tensor
    """
    result = to_tensor(
        erf(tensor),
        is_vector=tensor.is_vector,
        requires_grad=tensor.requires_grad,
    )
    SQRT_PI = 1.7724538509055159
    result.args = (tensor,)
    result.name = "erf"
    result.back_fns = (lambda x: (x * -2) / SQRT_PI,)

    return result
