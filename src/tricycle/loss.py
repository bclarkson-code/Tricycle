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
    REALLY_SMALL_NUMBER = 1e-8
    REALLY_BIG_NUMBER = 1e8

    def backward(self, grad: Tensor) -> Tensor:
        xp = grad.xp

        self._grad = xp.where(self._y_true == 1, -1 / self._y_pred, 0)
        self._grad *= xp.expand_dims(grad._data, -1)
        return to_tensor(self._grad, is_vector=grad.is_vector)

    def forward(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        # sourcery skip: assign-if-exp, reintroduce-else
        """
        Calculate the cross entropy loss
        """
        # normalise
        # TODO: fuse normalising and calculation together
        y_pred = Softmax()(y_pred)

        xp = y_pred.xp

        # clip for numeric stability
        y_pred._data = y_pred._data.clip(
            min=self.REALLY_SMALL_NUMBER, max=self.REALLY_BIG_NUMBER
        )

        # cache inputs for calculating the backwards operations later
        self._y_true = y_true._data
        self._y_pred = y_pred._data

        indicator = xp.where(y_true._data == 1, -xp.log(y_pred._data), 0)

        self._out = indicator.sum(axis=-1)

        result = to_tensor(self._out, is_vector=y_pred.is_vector)
        result.back_fns = (self.backward,)

        # y_true never requires grad so we dont calculate gradients for it
        result.args = (y_pred,)
        result.name = "cross_entropy"

        return result


class BinaryCrossEntropy(Op):
    """
    Calculate cross entropy loss, given logits and target indices (as opposed
    to one-hot encoded tensors)
    """

    REALLY_SMALL_NUMBER = 1e-8
    REALLY_BIG_NUMBER = 1e8

    def backward(self, grad: Tensor) -> Tensor:
        xp = grad.xp

        match self._y_pred.ndim:
            case 3:
                out = xp.zeros_like(self._y_pred)
                batch_indices = xp.arange(self._y_true.shape[0])
                token_indices = xp.arange(self._y_true.shape[1])
                for b in batch_indices:
                    out[b, token_indices, self._y_true[b]] = (
                        -1 / self._y_pred[b, token_indices, self._y_true[b]]
                    ) * grad._data[b]
            case 2:
                indices = xp.arange(self._y_true.shape[0])
                out = -1 / self._y_pred[indices, self._y_true._data]
                out *= grad._data
            case _:
                raise NotImplementedError(
                    "BinaryCrossEntropy with predictions with ndim: "
                    f"{self._y_pred.ndim} are not yet supported"
                )
        self._grad = out

        return to_tensor(self._grad, is_vector=grad.is_vector)

    def forward(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        # sourcery skip: assign-if-exp, reintroduce-else
        """
        Calculate the cross entropy loss
        """
        # normalise
        # TODO: fuse normalising and calculation together
        y_pred = Softmax()(y_pred)

        xp = y_pred.xp

        # clip for numeric stability
        y_pred._data = y_pred._data.clip(
            min=self.REALLY_SMALL_NUMBER, max=self.REALLY_BIG_NUMBER
        )

        # cache inputs for calculating the backwards operations later
        self._y_true = y_true._data
        self._y_pred = y_pred._data

        match self._y_pred.ndim:
            case 3:
                out = xp.zeros_like(y_true._data)
                batch_indices = xp.arange(y_true.shape[0])
                token_indices = xp.arange(y_true.shape[1])
                for b in batch_indices:
                    out[b] = -xp.log(
                        y_pred._data[b, token_indices, y_true._data[b]]
                    )
            case 2:
                indices = xp.arange(y_true.shape[0])
                out = -xp.log(y_pred[indices, y_true._data])
            case _:
                raise NotImplementedError(
                    "BinaryCrossEntropy with predictions with ndim: "
                    f"{self._y_pred.ndim} are not yet supported"
                )

        self._out = out

        result = to_tensor(self._out, is_vector=y_pred.is_vector)
        result.back_fns = (self.backward,)

        # y_true never requires grad so we dont calculate gradients for it
        result.args = (y_pred,)
        result.name = "cross_entropy"

        return result


class BinaryCrossEntropyV2(Op):
    """
    Calculate cross entropy loss, given logits and target indices (as opposed
    to one-hot encoded tensors)
    """

    REALLY_SMALL_NUMBER = 1e-8
    REALLY_BIG_NUMBER = 1e8

    def backward(self, grad: Tensor) -> Tensor:
        xp = grad.xp

        self._y_pred[xp.arange(self._n_inputs), self._y_true] -= 1
        self._y_pred /= self._n_inputs
        self._grad = self._y_pred.reshape(self._original_shape) * grad._data

        return to_tensor(self._grad, is_vector=self._input_vector)

    def forward(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        # sourcery skip: assign-if-exp, reintroduce-else
        """
        Calculate the cross entropy loss
        """
        xp = y_pred.xp

        # flatten to simplify multiple inputs
        self._original_shape = y_pred.shape
        self._input_vector = y_pred.is_vector
        out_dim = y_pred.shape[-1]
        self._y_true = y_true._data.reshape(-1)
        y_pred_f = y_pred._data.reshape(-1, out_dim)
        self._n_inputs = y_pred_f.shape[0]

        # we scale values by the largest value in each vector
        # for numeric stability
        max_vals = xp.max(y_pred_f, axis=-1, keepdims=True)
        scaled = y_pred_f - max_vals

        log_probs = scaled - xp.log(
            xp.sum(xp.exp(scaled), axis=-1, keepdims=True)
        )
        self._y_pred = xp.exp(log_probs)

        corrected_log_probs = -log_probs[
            xp.arange(self._n_inputs), self._y_true
        ]
        self._out = corrected_log_probs.sum() / self._n_inputs

        # TODO: fuse normalising and calculation together
        result = to_tensor(self._out, is_vector=False)
        result.back_fns = (self.backward,)

        # y_true never requires grad so we dont calculate gradients for it
        result.args = (y_pred,)
        result.name = "cross_entropy"

        return result
