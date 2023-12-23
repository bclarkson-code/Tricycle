from llm_from_scratch.ops import Tensor, log, mean, reduce_sum, softmax


def mean_square_error(y_pred: Tensor, y_true: Tensor) -> Tensor:
    """
    Copmute the mean square error between two identically shaped tensors
    """
    assert y_pred.shape == y_true.shape
    square_errors = (y_pred - y_true) ** 2
    return reduce_sum(square_errors) / y_pred.shape[0]


def categorical_crossentropy(y_pred: Tensor, y_true: Tensor) -> Tensor:
    """
    Compute the cross entropy loss between two identically shaped tensors.
    Note, y_pred are assumed to be logits. That is, they are expectected to
    be-unnormalised. y_true is assumed to be normalised already.
    Usually this is a one-hor encoding of a single categorical output
    but sharing labels across multiple outputs is possible.
    """
    y_pred = softmax(y_pred)
    y_pred = log(y_pred)

    loss = 0
    loss -= y_pred * y_true

    return mean(loss)
