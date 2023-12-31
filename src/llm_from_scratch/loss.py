from string import ascii_letters

from llm_from_scratch.ops import (Tensor, einsum, exp, log, max, mean,
                                  reduce_sum)


def mean_square_error(y_pred: Tensor, y_true: Tensor) -> Tensor:
    """
    Copmute the mean square error between two identically shaped tensors
    """
    assert y_pred.shape == y_true.shape
    errors = y_pred - y_true
    indices = ascii_letters[: len(errors.shape)]
    subscripts = f"{indices}, {indices} -> {indices}"
    square_errors = einsum(errors, errors, subscripts=subscripts)
    # print(f"{y_pred=}, {y_true=}, {errors=}, {square_errors=}")
    mse = mean(square_errors)
    # print(f"{mse=}")
    return mse


def categorical_crossentropy(y_pred: Tensor, y_true: Tensor) -> Tensor:
    """
    Compute the cross entropy loss between two identically shaped tensors.
    Note, y_pred are assumed to be logits. That is, they are expectected to
    be-unnormalised. y_true is assumed to be normalised already.
    Usually this is a one-hor encoding of a single categorical output
    but sharing labels across multiple outputs is possible.
    """
    # y_pred -= max(y_pred)
    coef = 1 / einsum(exp(y_pred), subscripts="ij->i")
    normalised_pred = einsum(coef, exp(y_pred), subscripts="i,ij->ij")
    log_probs = log(normalised_pred)
    loss = -einsum(log_probs, y_true, subscripts="ij,ij->i")
    return mean(loss)
