import logging

from tricycle.binary import BMul, _shapes_match
from tricycle.functions import softmax
from tricycle.tensor import Tensor, to_tensor
from tricycle.unary import ULog
from tricycle.utils import log_gpu_memory

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
    REALLY_SMALL_NUMBER = 1e-8
    # normalise
    y_pred = softmax(y_pred)

    xp = y_pred.xp
    y_pred._data = y_pred._data.clip(min=REALLY_SMALL_NUMBER, max=None)
    indicator = xp.where(y_true._data == 1, -xp.log(y_pred._data), 0)

    out = indicator.sum(axis=-1)

    def cross_entropy_back_fn(grad):
        result = xp.where(y_true._data == 1, -1 / y_pred._data, 0)
        return to_tensor(result, is_vector=grad.is_vector)

    out = to_tensor(out, is_vector=y_pred.is_vector)
    out.back_fns = (cross_entropy_back_fn,)
    # y_true never requires grad so we dont calculate gradients for it
    out.args = (y_pred,)
    out.name = "cross_entropy"

    return out


def cross_entropy_(y_true: Tensor, y_pred: Tensor) -> Tensor:
    # sourcery skip: assign-if-exp, reintroduce-else
    """
    Calculate the cross entropy loss
    """
    # normalise and log
    assert _shapes_match(y_true, y_pred)
    y_pred = ULog(softmax(y_pred))
    product = BMul(y_true, y_pred)
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
