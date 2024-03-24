from tricycle.binary import bdiv
from tricycle.reduce import rmax
from tricycle.tensor import Tensor
from tricycle.unary import udiv, uexp


def softmax(tensor):
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


def sigmoid(tensor: Tensor):
    """
    Apply the sigmoid function
    """
    return udiv(1, (uexp(-tensor) + 1))
