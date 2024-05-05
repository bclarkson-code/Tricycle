import logging

from tricycle.binary import _shapes_match, bmul
from tricycle.functions import softmax
from tricycle.tensor import Tensor
from tricycle.unary import ulog

logger = logging.getLogger(__name__)


def mean_square_error(y_true: Tensor, y_pred: Tensor):
    # sourcery skip: assign-if-exp, reintroduce-else
    square_error = (y_true - y_pred) ** 2
    assert isinstance(square_error, Tensor)

    divisor = square_error.shape[-1]
    if divisor == 1:
        return square_error

    return square_error.mean()


def cross_entropy(y_true: Tensor, y_pred: Tensor) -> Tensor:
    # sourcery skip: assign-if-exp, reintroduce-else
    """
    Calculate the cross entropy loss
    """
    # normalise and log
    assert _shapes_match(y_true, y_pred)
    y_pred = ulog(softmax(y_pred))
    product = bmul(y_true, y_pred)
    product *= -1

    match product.ndim:
        case 1:
            return product.e("...a->...")
        case 2:
            return product.e("...a->...")
        case 3:
            return product.e("...a->...")
        case 4:
            return product.e("...a->...")
