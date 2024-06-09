import logging

from tricycle.functions import Softmax
from tricycle.ops import Op
from tricycle.tensor import Tensor, to_tensor

logger = logging.getLogger(__name__)


def mean_square_error(y_true: Tensor, y_pred: Tensor):
    # sourcery skip: assign-if-exp, reintroduce-else
    square_error = (y_true - y_pred) ** 2
    assert isinstance(square_error, Tensor)

    divisor = square_error.shape[-1]
    if divisor == 1:
        return square_error

    return square_error.mean()


class CrossEntropy(Op):
    """
    Calculate cross entropy loss, given logits and target indices (as opposed
    to one-hot encoded tensors)
    """

    def log_softmax(self, tensor: Tensor):
        xp = tensor.xp
        x_max = xp.max(tensor.array, axis=-1, keepdims=True)
        log_sum_exp = x_max + xp.log(
            xp.sum(xp.exp(tensor.array - x_max), axis=-1, keepdims=True)
        )
        return tensor.array - log_sum_exp

    def forward(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        """
        Calculate the cross entropy loss
        """
        xp = y_pred.xp

        # Calculate log softmax
        log_softmax_pred = self.log_softmax(y_pred)

        # Cache for backward pass
        self._y_true = y_true.array
        self._log_softmax_pred = log_softmax_pred

        ndim = log_softmax_pred.ndim

        if ndim == 3:
            batch_indices = xp.arange(y_true.shape[0], dtype=int)
            token_indices = xp.arange(y_true.shape[1], dtype=int)
            loss = -log_softmax_pred[
                batch_indices[:, None], token_indices, y_true.array
            ]
        elif ndim == 2:
            indices = xp.arange(y_true.shape[0], dtype=int)
            loss = -log_softmax_pred[indices, y_true.array]
        elif ndim == 1:
            loss = -log_softmax_pred[y_true.array]
        else:
            raise NotImplementedError(
                f"BinaryCrossEntropy with predictions with ndim: {ndim} are not yet supported"
            )

        # Mean loss over all elements
        loss = loss.mean()

        self._out = loss
        result = to_tensor(self._out, is_batched=False)
        result.back_fns = (self.backward,)

        result.args = (y_pred,)
        result.name = "cross_entropy"

        return result

    def backward(self, grad: Tensor) -> Tensor:
        xp = grad.xp
        ndim = self._log_softmax_pred.ndim

        if ndim == 3:
            batch_indices = xp.arange(self._y_true.shape[0], dtype=int)
            token_indices = xp.arange(self._y_true.shape[1], dtype=int)
            grad_output = xp.exp(self._log_softmax_pred)
            grad_output[
                batch_indices[:, None], token_indices, self._y_true
            ] -= 1
            grad_output *= grad.array / (
                self._y_true.shape[0] * self._y_true.shape[1]
            )

        elif ndim == 2:
            indices = xp.arange(self._y_true.shape[0], dtype=int)
            grad_output = xp.exp(self._log_softmax_pred)
            grad_output[indices, self._y_true] -= 1
            grad_output *= grad.array / self._y_true.shape[0]
        elif ndim == 1:
            grad_output = xp.exp(self._log_softmax_pred)
            grad_output[self._y_true] -= 1
            grad_output *= grad.array
        else:
            raise NotImplementedError(
                f"BinaryCrossEntropy with predictions with ndim: {ndim} are not yet supported"
            )

        self._grad = grad_output
        return to_tensor(self._grad, is_batched=grad.is_batched)
