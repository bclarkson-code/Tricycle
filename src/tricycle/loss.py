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


class CrossEntropy_(Op):
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


class CrossEntropy(Op):
    """
    Calculate cross entropy loss, given logits and target indices (as opposed
    to one-hot encoded tensors)
    """

    def log_softmax(self, tensor: Tensor):
        xp = tensor.xp
        x_max = xp.max(tensor._data, axis=-1, keepdims=True)
        log_sum_exp = x_max + xp.log(
            xp.sum(xp.exp(tensor._data - x_max), axis=-1, keepdims=True)
        )
        return tensor._data - log_sum_exp

    def forward(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        """
        Calculate the cross entropy loss
        """
        xp = y_pred.xp

        # Calculate log softmax
        log_softmax_pred = self.log_softmax(y_pred)

        # Cache for backward pass
        self._y_true = y_true._data
        self._log_softmax_pred = log_softmax_pred

        ndim = log_softmax_pred.ndim

        if ndim == 3:
            batch_indices = xp.arange(y_true.shape[0], dtype=int)
            token_indices = xp.arange(y_true.shape[1], dtype=int)
            loss = -log_softmax_pred[
                batch_indices[:, None], token_indices, y_true._data
            ]
        elif ndim == 2:
            indices = xp.arange(y_true.shape[0], dtype=int)
            loss = -log_softmax_pred[indices, y_true._data]
        elif ndim == 1:
            loss = -log_softmax_pred[y_true._data]
        else:
            raise NotImplementedError(
                f"BinaryCrossEntropy with predictions with ndim: {ndim} are not yet supported"
            )

        # Mean loss over all elements
        loss = loss.mean()

        self._out = loss
        result = to_tensor(self._out, is_vector=False)
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
            grad_output *= grad._data / (
                self._y_true.shape[0] * self._y_true.shape[1]
            )

        elif ndim == 2:
            indices = xp.arange(self._y_true.shape[0], dtype=int)
            grad_output = xp.exp(self._log_softmax_pred)
            grad_output[indices, self._y_true] -= 1
            grad_output *= grad._data / self._y_true.shape[0]
        elif ndim == 1:
            grad_output = xp.exp(self._log_softmax_pred)
            grad_output[self._y_true] -= 1
            grad_output *= grad._data
        else:
            raise NotImplementedError(
                f"BinaryCrossEntropy with predictions with ndim: {ndim} are not yet supported"
            )

        self._grad = grad_output
        return to_tensor(self._grad, is_vector=grad.is_vector)
