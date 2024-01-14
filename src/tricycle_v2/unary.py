from functools import partial
from string import ascii_letters
from typing import Union

import numpy as np

from tricycle_v2.ops import einsum, nothing, to_tensor
from tricycle_v2.tensor import Tensor

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
    indices = ascii_letters[: len(tensor.shape)]
    subscripts = f"{indices},{indices}->{indices}"
    return einsum(subscripts, tensor, constant_tensor)


def usub(arg_1: Union[Tensor, float], arg_2: Union[Tensor, float]) -> Tensor:
    """
    Subtract a constant, elementwise, from a tensor. The constant is not
    differentiable.
    """
    if isinstance(arg_1, Tensor) and np.isscalar(arg_2):
        return uadd(arg_1, -arg_2)
    elif isinstance(arg_2, Tensor) and np.isscalar(arg_1):
        return uadd(umul(arg_2, -1), arg_1)
    else:
        raise NotImplementedError(
            f"Subtraction between {type(arg_1)} and {type(arg_2)}"
        )


def upow(tensor: Tensor, constant: float) -> Tensor:
    """
    Raise a tensor to a constant, elementwise. The constant is not
    differentiable.
    """
    assert isinstance(tensor, Tensor)
    assert np.isscalar(constant)

    result = to_tensor(np.power(tensor, constant))
    result.args = (tensor,)

    coeff = to_tensor(np.power(tensor, constant - 1))
    coeff = umul(coeff, constant)

    assert coeff.shape == tensor.shape
    indices = ascii_letters[: len(tensor.shape)]
    subscripts = f"{indices},{indices}->{indices}"

    result.back_fn = (partial(einsum, subscripts, coeff),)
    return result


def udiv(arg_1: Union[Tensor, float], arg_2: Union[Tensor, float]) -> Tensor:
    """
    Divide a tensor by a constant, elementwise. The constant is not
    differentiable.
    """
    if isinstance(arg_1, Tensor) and np.isscalar(arg_2):
        return umul(arg_1, 1 / arg_2)

    elif isinstance(arg_2, Tensor) and np.isscalar(arg_1):
        return umul(upow(arg_2, -1.0), arg_1)
    else:
        raise NotImplementedError(f"Division between {type(arg_1)} and {type(arg_2)}")


def umax(tensor: Tensor, constant: float) -> Tensor:
    """
    Max a tensor by a constant, elementwise. The constant is not
    differentiable.
    """
    assert isinstance(tensor, Tensor)
    assert np.isscalar(constant)

    result = to_tensor(np.maximum(tensor, constant))

    indicator = to_tensor((tensor > constant).astype(float))
    indices = ascii_letters[: len(tensor.shape)]
    subscripts = f"{indices},{indices}->{indices}"

    result.args = (tensor,)
    result.back_fn = (partial(einsum, subscripts, indicator),)
    return result


def umin(tensor: Tensor, constant: float) -> Tensor:
    """
    Min a tensor by a constant, elementwise. The constant is not
    differentiable.
    """
    assert isinstance(tensor, Tensor)
    assert np.isscalar(constant)

    result = to_tensor(np.minimum(tensor, constant))

    indicator = to_tensor((tensor <= constant).astype(float))
    indices = ascii_letters[: len(tensor.shape)]
    subscripts = f"{indices},{indices}->{indices}"

    result.args = (tensor,)
    result.back_fn = (partial(einsum, subscripts, indicator),)
    return result


def uexp(tensor: Tensor) -> Tensor:
    """
    Raise every element of a tensor to the power of e
    """
    result = to_tensor(np.exp(tensor))

    indices = ascii_letters[: len(tensor.shape)]
    subscripts = f"{indices},{indices}->{indices}"

    result.back_fn = (partial(einsum, subscripts, result),)
    result.args = (tensor,)
    return result


def ulog(tensor: Tensor) -> Tensor:
    """
    Raise every element of a tensor to the power of e
    """
    result = to_tensor(np.log(tensor))

    indices = ascii_letters[: len(tensor.shape)]
    subscripts = f"{indices},{indices}->{indices}"

    coeff = udiv(1, tensor)

    result.back_fn = (partial(einsum, subscripts, coeff),)
    result.args = (tensor,)

    return result


def usin(tensor: Tensor) -> Tensor:
    """
    Raise every element of a tensor to the power of e
    """
    result = to_tensor(np.sin(tensor))

    indices = ascii_letters[: len(tensor.shape)]
    subscripts = f"{indices},{indices}->{indices}"

    coef = to_tensor(np.cos(tensor))

    result.back_fn = (partial(einsum, subscripts, coef),)
    result.args = (tensor,)

    return result


def ucos(tensor: Tensor) -> Tensor:
    """
    Raise every element of a tensor to the power of e
    """
    result = to_tensor(np.cos(tensor))

    indices = ascii_letters[: len(tensor.shape)]
    subscripts = f"{indices},{indices}->{indices}"

    coef = to_tensor(-np.sin(tensor))

    result.back_fn = (partial(einsum, subscripts, coef),)
    result.args = (tensor,)

    return result
