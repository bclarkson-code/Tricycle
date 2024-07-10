from tricycle import TRICYCLE_CONTEXT
from tricycle.tensor import Tensor, to_tensor


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
    ):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.momentum = momentum

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
            tensor.array.grad = tensor.array.grad.astype(xp.float32)

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

        out = tensor.array - grad
        if TRICYCLE_CONTEXT.use_mixed_precision:
            out = out.astype(xp.float16)

        # update the value only, leave everything else
        result = Tensor(
            out,
            requires_grad=tensor.requires_grad,
            name=tensor.name,
            is_batched=tensor.is_batched,
            _id=tensor._id,
        )

        assert result.shape == tensor.shape

        # TODO: figure out whether these can be safely removed
        del tensor
        del grad

        return result

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
        self.timestep = 0

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
            grad = grad.astype(xp.float32)
            tensor.array = tensor.array.astype(xp.float32)

        # initialise stores
        if key not in self.momentum:
            self.momentum[key] = xp.zeros_like(grad, dtype=grad.dtype)
        if key not in self.square_momentum:
            self.square_momentum[key] = tensor.xp.zeros_like(
                grad, dtype=grad.dtype
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

        tensor.array -= self.learning_rate * (
            momentum_estimate / (xp.sqrt(square_momentum_estimate) + self.eps)
            + self.weight_decay * tensor.array
        )

        tensor.grad.array.fill(0)
        if TRICYCLE_CONTEXT.use_mixed_precision:
            tensor.grad.array = tensor.grad.array.astype(xp.float16)
            tensor.array = tensor.array.astype(xp.float16)

        return tensor

    def __call__(self, tensor: Tensor) -> Tensor:
        return self._reset_grad(self.update_weight(tensor))
