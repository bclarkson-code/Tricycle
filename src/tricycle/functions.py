import numpy as np
from numba import njit
from scipy.special import softmax as scipy_softmax

from tricycle.binary import bdiv
from tricycle.einsum import Einsum
from tricycle.reduce import rmax
from tricycle.tensor import Tensor, to_tensor
from tricycle.unary import udiv, uexp


def softmax(tensor: Tensor):
    """
    Apply softmax. The softmax is only applied to the final
    dimension of the tensor
    Note: the tensor is normalised for numeric stability
    """

    # normalise
    largest_element = rmax(tensor, "...a->...").repeat(tensor.shape[-1])
    tensor = tensor - largest_element

    numerator = uexp(tensor)
    denominator = numerator.e("...a->...").repeat(tensor.shape[-1])
    return bdiv(numerator, denominator)


def softmax_v2(tensor: Tensor):
    """
    Apply softmax. The softmax is only applied to the final
    dimension of the tensor
    Note: the tensor is normalised for numeric stability
    """
    if tensor.on_gpu:
        from cupyx.scipy.special import softmax as softmax_fn
    else:
        from scipy.special import softmax as softmax_fn

    _result = softmax_fn(tensor._data, axis=-1)

    def softmax_back_fn(grad: Tensor):
        xp = grad.xp
        match _result.ndim:
            case 1:
                n_elements = _result.shape[0]
                diag = xp.identity(n_elements)

                left = xp.einsum("i,ij,i->j", _result, diag, grad._data)
                right = xp.einsum("i,j,i->j", _result, _result, grad._data)
            case 2:
                batch_size, n_elements = _result.shape
                diag = xp.tile(xp.identity(n_elements), batch_size).T
                diag = diag.reshape(batch_size, n_elements, n_elements)

                left = xp.einsum("ki,kij,ki->kj", _result, diag, grad._data)
                right = xp.einsum("ki,kj,ki->kj", _result, _result, grad._data)
            case 3:
                batch_size, n_tokens, n_elements = _result.shape
                diag = xp.tile(xp.identity(n_elements), n_tokens)
                diag = xp.tile(diag, batch_size).T
                diag = diag.reshape(
                    batch_size, n_tokens, n_elements, n_elements
                )

                left = xp.einsum(
                    "zki,zkij,zki->zkj", _result, diag, grad._data
                )
                right = xp.einsum(
                    "zki,zkj,zki->zkj", _result, _result, grad._data
                )
            case _:
                raise NotImplementedError()

        output = left - right
        return to_tensor(output, is_vector=grad.is_vector, name="back_softmax")

    result = to_tensor(_result)
    result.args = (tensor,)
    result.name = "softmax"
    result.is_vector = tensor.is_vector
    result.back_fns = (softmax_back_fn,)

    return result


def softmax_v3(tensor: Tensor):
    """
    Apply softmax. The softmax is only applied to the final
    dimension of the tensor
    Note: the tensor is normalised for numeric stability
    """
    if tensor.on_gpu:
        from cupyx.scipy.special import softmax as softmax_fn
    else:
        from scipy.special import softmax as softmax_fn

    _result = softmax_fn(tensor._data, axis=-1)

    def softmax_back_fn(grad: Tensor):
        xp = grad.xp
        match _result.ndim:
            case 1:
                out = xp.zeros(_result.shape)
                for i in range(_result.shape[0]):
                    for j in range(_result.shape[0]):
                        is_diagonal = i == j
                        local_derivative = _result[i] * (
                            is_diagonal - _result[j]
                        )
                        out[j] += local_derivative * grad._data[i]
            case 2:
                out = xp.zeros(_result.shape)
                for t in range(_result.shape[0]):
                    for i in range(_result.shape[1]):
                        for j in range(_result.shape[1]):
                            is_diagonal = i == j
                            local_derivative = _result[t][i] * (
                                is_diagonal - _result[t][j]
                            )
                            out[t][j] += local_derivative * grad._data[t][i]
            case 3:
                out = xp.zeros(_result.shape)
                for z in range(_result.shape[0]):
                    for t in range(_result.shape[1]):
                        for i in range(_result.shape[2]):
                            for j in range(_result.shape[2]):
                                is_diagonal = i == j
                                local_derivative = _result[z][t][i] * (
                                    is_diagonal - _result[z][t][j]
                                )
                                out[z][t][j] += (
                                    local_derivative * grad._data[z][t][i]
                                )
            case _:
                raise NotImplementedError()

        return to_tensor(out, is_vector=grad.is_vector, name="back_softmax")

    result = to_tensor(_result)
    result.args = (tensor,)
    result.name = "softmax"
    result.is_vector = tensor.is_vector
    result.back_fns = (softmax_back_fn,)

    return result


def sigmoid(tensor: Tensor):
    """
    Apply the sigmoid function
    """
    return udiv(1, (uexp(-tensor) + 1))


def tanh(tensor: Tensor):
    """
    Apply the tanh function
    """
    numerator = uexp(tensor * 2) - 1
    denominator = uexp(tensor * 2) + 1
    return bdiv(numerator, denominator)
