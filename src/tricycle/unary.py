import numbers
from functools import partial

from scipy.special import erf as np_erf

from tricycle import CUPY_ENABLED
from tricycle.einsum import Einsum
from tricycle.tensor import Tensor, nothing, to_tensor

grad = False


def uadd(tensor: Tensor, constant: float) -> Tensor:
    """
    Add a constant, elementwise, to a tensor. The constant is not
    differentiable.
    """
    xp = tensor.xp

    assert isinstance(tensor, Tensor)
    assert isinstance(constant, numbers.Number)

    result = to_tensor(xp.add(tensor._data, constant, dtype=tensor.dtype))
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
    xp = tensor.xp

    assert isinstance(tensor, Tensor)
    assert xp.isscalar(constant)

    constant_tensor = to_tensor(
        xp.full(shape=tensor.shape, fill_value=constant, dtype=tensor.dtype),
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
    xp = tensor.xp

    from tricycle.binary import bmul

    assert isinstance(tensor, Tensor)
    assert xp.isscalar(constant)

    result = to_tensor(xp.power(tensor._data, constant))
    result.args = (tensor,)
    coef = to_tensor(
        xp.power(tensor._data, constant - 1, dtype=tensor.dtype),
        is_vector=tensor.is_vector,
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
    xp = tensor.xp

    assert isinstance(tensor, Tensor)
    assert xp.isscalar(constant)

    result = to_tensor(xp.maximum(tensor._data, constant, dtype=tensor.dtype))

    from tricycle.binary import bmul

    result.args = (tensor,)
    is_bigger = tensor > constant
    is_bigger.is_vector = tensor.is_vector
    result.back_fns = (partial(bmul, is_bigger),)
    result.name = f"> {constant}"
    result.is_vector = tensor.is_vector

    return result


def umin(tensor: Tensor, constant: float) -> Tensor:
    """
    Return the max of the tensor and the constant,
    elementwise. The constant is not differentiable.
    """
    xp = tensor.xp

    assert isinstance(tensor, Tensor)
    assert xp.isscalar(constant)

    result = to_tensor(xp.minimum(tensor._data, constant, dtype=tensor.dtype))

    from tricycle.binary import bmul

    result.args = (tensor,)
    is_smaller = tensor < constant
    is_smaller.is_vector = tensor.is_vector
    result.back_fns = (partial(bmul, is_smaller),)
    result.name = f"< {constant}"
    result.is_vector = tensor.is_vector
    return result


def uexp(tensor: Tensor) -> Tensor:
    """
    Raise every element of a tensor to the power of e
    """
    xp = tensor.xp

    result = to_tensor(xp.exp(tensor._data))

    from tricycle.binary import bmul

    result.args = (tensor,)
    result.name = "exp"
    result.is_vector = tensor.is_vector
    coef = to_tensor(result._data, is_vector=tensor.is_vector)
    result.back_fns = (partial(bmul, coef),)
    return result


def ulog(tensor: Tensor) -> Tensor:
    """
    Raise every element of a tensor to the power of e
    """
    REALLY_SMALL_NUMBER = 1e-8
    xp = tensor.xp

    result = to_tensor(xp.log(tensor._data))

    from tricycle.binary import bmul

    result.args = (tensor,)
    result.back_fns = (
        partial(
            bmul,
            udiv(1, tensor + REALLY_SMALL_NUMBER),
        ),
    )
    result.name = "log"
    result.is_vector = tensor.is_vector
    return result


def usin(tensor: Tensor) -> Tensor:
    """
    Raise every element of a tensor to the power of e
    """
    xp = tensor.xp
    result = to_tensor(xp.sin(tensor._data))

    from tricycle.binary import bmul

    result.args = (tensor,)
    coef = to_tensor(xp.cos(tensor._data), is_vector=tensor.is_vector)
    result.back_fns = (partial(bmul, coef),)
    result.name = "sin"
    result.is_vector = tensor.is_vector
    return result


def ucos(tensor: Tensor) -> Tensor:
    """
    Raise every element of a tensor to the power of e
    """
    xp = tensor.xp
    result = to_tensor(xp.cos(tensor._data))

    from tricycle.binary import bmul

    result.args = (tensor,)
    coef = to_tensor(-xp.sin(tensor._data), is_vector=tensor.is_vector)
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
    if CUPY_ENABLED:
        import cupy
        from cupyx.scipy.special import erf as cp_erf

        erf = cp_erf if isinstance(tensor._data, cupy.ndarray) else np_erf
    else:
        erf = np_erf

    result = to_tensor(
        erf(tensor._data),
        is_vector=tensor.is_vector,
        requires_grad=tensor.requires_grad,
    )
    SQRT_PI = 1.7724538509055159
    result.args = (tensor,)
    result.name = "erf"
    result.back_fns = (lambda x: (x * -2) / SQRT_PI,)

    return result


def usum(tensor: Tensor) -> Tensor:
    """
    Sum all the elements of a tensor into a single value
    """
    xp = tensor.xp
    if tensor.is_vector:
        raise NotImplementedError(
            "Sum is not yet implemented for vectorised tensors"
        )
    result = to_tensor(xp.sum(tensor._data))
    result.args = (tensor,)

    def sum_back_fn(grad):
        xp = grad.xp
        result = xp.full_like(tensor._data, grad._data)
        return to_tensor(result)

    result.back_fns = (sum_back_fn,)
    result.name = "sum"
    result.is_vector = False

    return result
