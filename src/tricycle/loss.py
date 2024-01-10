from string import ascii_letters

from tricycle.ops import Tensor, einsum, tensor, log, mean, softmax


def mean_square_error(y_pred: Tensor, y_true: Tensor) -> Tensor:
    """
    Copmute the mean square error between two identically shaped tensors
    """
    assert y_pred.shape == y_true.shape

    errors = y_pred - y_true

    indices = ascii_letters[: len(errors.shape)]
    subscripts = f"{indices}, {indices} -> {indices}"

    square_errors = einsum(errors, errors, subscripts=subscripts)
    return mean(square_errors)


def categorical_crossentropy(y_logits: Tensor, y_true: Tensor) -> Tensor:
    """
    NOTE: This does not currently work. I cannot get the test to pass

    Compute the cross entropy loss between two identically shaped tensors.
    Note, y_pred are assumed to be logits. That is, they are expectected to
    be-unnormalised. y_true is assumed to be normalised already.
    Usually this is a one-hor encoding of a single categorical output
    but sharing labels across multiple outputs is possible.
    """
    y_pred = softmax(y_logits)
    y_pred.name = "y_pred"
    log_probs = log(y_pred)
    log_probs.name = "log_probs"
    loss = -einsum(log_probs, y_true, subscripts="ij,ij->i")
    loss.name = "loss"
    # raise Exception(f"{log_probs=}, {y_true=}, {y_pred=}, {y_true=}, {loss=}")
    return einsum(loss, subscripts="i->") / tensor(loss.shape[0], requires_grad=False)
