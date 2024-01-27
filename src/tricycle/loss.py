from string import ascii_lowercase

from tricycle.binary import bmul
from tricycle.ops import softmax
from tricycle.reduce import radd
from tricycle.tensor import Tensor
from tricycle.unary import ulog, umul


def mean_squared_error(y_true: Tensor, y_pred: Tensor) -> Tensor:
    """
    Calcuate the mean square error along the final index of a tensor
    """
    square_error = (y_true - y_pred) ** 2
    return radd(square_error, "i->")


def cross_entropy(y_true: Tensor, y_pred: Tensor) -> Tensor:
    """
    Calculate the cross entropy loss
    """
    # normalise and log
    y_pred = ulog(softmax(y_pred))
    return umul(radd(bmul(y_true, y_pred), "i->"), -1)
