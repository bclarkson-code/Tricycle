from tricycle.ops import softmax
from tricycle.tensor import Tensor
from tricycle.unary import ulog


def mean_square_error(y_true: Tensor, y_pred: Tensor):
    square_error = (y_true - y_pred) ** 2
    assert isinstance(square_error, Tensor)
    return square_error.e("i->")


def cross_entropy(y_true: Tensor, y_pred: Tensor) -> Tensor:
    """
    Calculate the cross entropy loss
    """
    # normalise and log
    y_pred = ulog(softmax(y_pred))
    product = y_true * y_pred * -1
    return product.e("i->")
