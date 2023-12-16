from llm_from_scratch.ops import Tensor, reduce_sum


def mean_square_error(y_pred: Tensor, y_true: Tensor) -> Tensor:
    """
    Copmute the mean square error between two identically shaped tensors
    """
    assert y_pred.shape == y_true.shape
    square_errors = (y_pred - y_true) ** 2
    return reduce_sum(square_errors) / y_pred.shape[0]
