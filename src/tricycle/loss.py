from string import ascii_lowercase

from tricycle.binary import bsub
from tricycle.reduce import radd
from tricycle.tensor import Tensor


def mean_squared_error(y_true: Tensor, y_pred: Tensor) -> Tensor:
    """
    Calcuate the mean square error along the final index of a tensor
    """
    square_error = (y_true - y_pred) ** 2
    indices = ascii_lowercase[: len(square_error.shape)]
    subscript = f"{indices}->{indices[:-1]}"
    total_error = radd(square_error, subscript)
    return total_error / square_error.shape[-1]
