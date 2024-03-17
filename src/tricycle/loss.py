import numpy as np

from tricycle.ops import softmax
from tricycle.tensor import Tensor, to_tensor
from tricycle.unary import ulog


def mean_square_error(y_true: Tensor, y_pred: Tensor):
    # sourcery skip: assign-if-exp, reintroduce-else
    square_error = (y_true - y_pred) ** 2
    assert isinstance(square_error, Tensor)

    divisor = square_error.shape[-1]
    if divisor == 1:
        return square_error

    result = square_error.e("i->")
    return result / divisor


def cross_entropy(y_true: Tensor, y_pred: Tensor) -> Tensor:
    # sourcery skip: assign-if-exp, reintroduce-else
    """
    Calculate the cross entropy loss
    """
    # normalise and log
    y_pred = ulog(softmax(y_pred))
    product = y_true * y_pred * -1

    return product.e("i->")
