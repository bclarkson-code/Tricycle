import logging

from tricycle import TRICYCLE_CONTEXT
from tricycle.ops import Op
from tricycle.tensor import Tensor

logger = logging.getLogger(__name__)


class MeanSquaredError(Op):
    """
    Calculate Mean Squared Error loss
    """

    def backward(self, grad: Tensor) -> Tensor:
        xp = grad.xp

        if TRICYCLE_CONTEXT.use_mixed_precision:
            grad.array = grad.array.astype(xp.float32)

        out = self.diff * 2 * grad.array * self.divisor

        if TRICYCLE_CONTEXT.use_mixed_precision:
            out = out.astype(xp.float16)

        return Tensor(out)

    def forward(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        xp = y_pred.xp

        if TRICYCLE_CONTEXT.use_mixed_precision:
            y_pred.array = y_pred.array.astype(xp.float32)
            y_true.array = y_true.array.astype(xp.float32)

        self.diff = y_pred.array - y_true.array
        self.divisor = 1 / xp.prod(y_pred.shape[-1])

        out = (self.diff**2).sum() * self.divisor

        if TRICYCLE_CONTEXT.use_mixed_precision:
            out *= TRICYCLE_CONTEXT.loss_scale_factor
            out = out.astype(xp.float16)

        if not xp.isfinite(out):
            raise ValueError("Loss is infinite")

        # only y_pred is differentiable: y_true is a constant
        return Tensor(
            out,
            args=(y_pred,),
            back_fns=(self.backward,),
            name="mean_squared_error",
        )


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
        # cross entropy reduces a huge matrix to a single number which makes
        # it really sensitive to errors. To rememdy this, we need to use
        # full precision
        if TRICYCLE_CONTEXT.use_mixed_precision:
            y_pred.array = y_pred.array.astype(xp.float32)

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
        if TRICYCLE_CONTEXT.use_mixed_precision:
            self._out = (
                self._out.astype(xp.float16)
                * TRICYCLE_CONTEXT.loss_scale_factor
            )

        return Tensor(
            self._out,
            is_batched=False,
            back_fns=(self.backward,),
            args=(y_pred,),
            name="cross_entropy",
        )

    def backward(self, grad: Tensor) -> Tensor:
        xp = grad.xp
        ndim = self._log_softmax_pred.ndim

        if TRICYCLE_CONTEXT.use_mixed_precision:
            grad.array = grad.array.astype(xp.float32)

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

        # remember to convert the gradient back to the right precision
        if TRICYCLE_CONTEXT.use_mixed_precision:
            self._grad = self._grad.astype(xp.float16)

        return Tensor(self._grad, is_batched=grad.is_batched)
