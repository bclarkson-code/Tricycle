from warnings import warn

from tricycle import TRICYCLE_CONTEXT
from tricycle.tensor import Tensor


class Optimiser:
    def __call__(self, tensor: Tensor) -> Tensor:
        raise NotImplementedError

    def _reset_grad(self, tensor: Tensor):
        tensor.grad = None
        tensor.args = None
        tensor.back_fns = None
        return tensor


class StochasticGradientDescent(Optimiser):
    def __init__(
        self,
        learning_rate: float,
        weight_decay: float | None = None,
        momentum: float | None = None,
        logger=None,
    ):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.logger = logger

        self.momentum_store = {}

    def update_weight(self, tensor: Tensor):
        """
        Perform a gradient update on a tensor, optionally
        including weight decay and momentum
        """
        xp = tensor.xp
        assert tensor.grad is not None

        # We need to do gradient updates in full precision otherwise we get
        # stability issues
        if TRICYCLE_CONTEXT.use_mixed_precision:
            tensor.grad.array = (
                tensor.grad.array.astype(xp.float32)
                / TRICYCLE_CONTEXT.loss_scale_factor
            )
            if not tensor.array.dtype == xp.float32:
                tensor.array = tensor.array.astype(xp.float32)

        if tensor.grad.is_batched:
            tensor.grad = tensor.grad.from_batched().einsum("z...->...")

        grad = self.learning_rate * tensor.grad.array

        if self.weight_decay is not None:
            wd = self.learning_rate * self.weight_decay * tensor.array
            grad += wd

        if self.momentum is not None and self.momentum > 0:
            if tensor._id not in self.momentum_store:
                last_momentum = tensor.xp.zeros(grad.shape)
            else:
                last_momentum = self.momentum_store[tensor._id]

            grad += self.momentum * last_momentum
            self.momentum_store[tensor._id] = grad

        # make sure our gradients aren't underflowing or overflow
        if not xp.isfinite(grad).all():
            warn(
                "Found nans in gradient, skipping this gradient and"
                "decreasing loss scaling. If this warning persists, "
                "check that your learning rate isn't too high"
            )
            TRICYCLE_CONTEXT.loss_scale_factor /= 2
            self.logger.error(
                f"New scaling factor: {TRICYCLE_CONTEXT.loss_scale_factor}"
            )
            return tensor

        if (grad == 0).sum() > grad.size * 0.05:
            warn(
                "Found too many 0's in gradient, skipping this gradient and"
                "increasing loss scaling. If this warning persists, "
                "check that your learning rate isn't too low"
            )
            TRICYCLE_CONTEXT.loss_scale_factor *= 2
            self.logger.error(
                f"New scaling factor: {TRICYCLE_CONTEXT.loss_scale_factor}"
            )
            return tensor

        if TRICYCLE_CONTEXT.use_mixed_precision:
            tensor.array -= grad.astype(xp.float32)

        tensor.grad.array.fill(0)
        return tensor

    def __call__(self, tensor: Tensor) -> Tensor:
        return self._reset_grad(self.update_weight(tensor))


class AdamW(Optimiser):
    def __init__(
        self,
        learning_rate=1e-3,
        betas=(0.9, 0.999),
        eps=1e-6,
        weight_decay=0.01,
    ):
        self.learning_rate = learning_rate
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.timestep = 1

        self.momentum = {}
        self.square_momentum = {}

    def step(self):
        # we compute the updates dynamically so we'll need to remember to
        # call this
        self.timestep += 1

    def update_weight(self, tensor: Tensor) -> Tensor:
        key = tensor._id
        xp = tensor.xp

        assert tensor.grad is not None
        grad = tensor.grad.array

        if TRICYCLE_CONTEXT.use_mixed_precision:
            grad = grad.astype(xp.float32) / TRICYCLE_CONTEXT.loss_scale_factor
            if not tensor.array.dtype == xp.float32:
                tensor.array = tensor.array.astype(xp.float32)

        # initialise stores
        if key not in self.momentum:
            self.momentum[key] = xp.zeros_like(grad, dtype=xp.float32)
        if key not in self.square_momentum:
            self.square_momentum[key] = tensor.xp.zeros_like(
                grad, dtype=xp.float32
            )

        self.momentum[key] = (
            self.betas[0] * self.momentum[key] + (1 - self.betas[0]) * grad
        )

        self.square_momentum[key] = self.betas[1] * self.square_momentum[
            key
        ] + (1 - self.betas[1]) * (grad * grad)

        momentum_estimate = self.momentum[key] / (
            1 - self.betas[0] ** self.timestep
        )
        square_momentum_estimate = self.square_momentum[key] / (
            1 - self.betas[1] ** self.timestep
        )

        combined_grad = self.learning_rate * (
            momentum_estimate / (xp.sqrt(square_momentum_estimate) + self.eps)
            + self.weight_decay * tensor.array
        )

        if not xp.isfinite(tensor.array).all():
            breakpoint()

        if (combined_grad == 0).sum() > combined_grad.size * 0.05:
            breakpoint()

        if TRICYCLE_CONTEXT.use_mixed_precision:
            tensor.array -= combined_grad.astype(xp.float32)

        tensor.grad.array.fill(0)
        return tensor

    def __call__(self, tensor: Tensor) -> Tensor:
        return self._reset_grad(self.update_weight(tensor))
