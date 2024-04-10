import numpy as np
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
        from cupy.scipy.special import softmax as softmax_fn
    else:
        from scipy.special import softmax as softmax_fn

    result = to_tensor(softmax_fn(tensor._data))
    result.args = (tensor,)
    result.name = "softmax"
    result.is_vector = tensor.is_vector

    def softmax_back_fn(grad: Tensor):
        diag = grad.xp.eye(tensor.shape[-1])
        left = Einsum("i,jk->ij")(grad, to_tensor(diag))
        right = Einsum("i,j->ij")(grad, grad)
        return left - right

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
