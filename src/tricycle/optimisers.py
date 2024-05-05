import numpy as np

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
        assert tensor.grad is not None

        if tensor.grad.is_vector:
            tensor.grad = tensor.grad.from_vector().e("z...->...")

        grad = self.learning_rate * tensor.grad

        if self.weight_decay is not None:
            wd = self.learning_rate * self.weight_decay * tensor
            grad += to_tensor(wd, name=f"weight_decay({self.weight_decay})")

        if self.momentum is not None and self.momentum > 0:
            if tensor._id not in self.momentum_store:
                last_momentum = to_tensor(tensor.xp.zeros(grad.shape))
            else:
                last_momentum = self.momentum_store[tensor._id]

            grad += self.momentum * last_momentum
            self.momentum_store[tensor._id] = to_tensor(grad._data)

        # update the value only, leave everything else
        result = to_tensor(
            tensor - grad,
            requires_grad=tensor.requires_grad,
            name=tensor.name,
            is_vector=tensor.is_vector,
            _id=tensor._id,
        )

        assert result.shape == tensor.shape

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
        eps=1e-8,
        weight_decay=0.01,
    ):
        self.learning_rate = learning_rate
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        # we compute the updates dynamically so we'll need to remember to
        # call this
        self.t += 1

    def update_weight(self, tensor: Tensor) -> Tensor:
        key = tensor._id
        xp = tensor.xp

        # initialise stores
        if key not in self.m:
            self.m[key] = xp.zeros_like(tensor._data)
        if key not in self.v:
            self.v[key] = tensor.xp.zeros_like(tensor._data)

        self.m[key] = (
            self.betas[0] * self.m[key]
            + (1 - self.betas[0]) * tensor.grad._data
        )

        self.v[key] = self.betas[1] * self.v[key] + (1 - self.betas[1]) * (
            tensor.grad._data**2
        )

        m_hat = self.m[key] / (1 - self.betas[0] ** self.t)
        v_hat = self.v[key] / (1 - self.betas[1] ** self.t)

        result = tensor._data - self.learning_rate * (
            m_hat / (xp.sqrt(v_hat) + self.eps)
            + self.weight_decay * tensor._data
        )
        result = to_tensor(
            result,
            requires_grad=tensor.requires_grad,
            name=tensor.name,
            is_vector=tensor.is_vector,
            _id=tensor._id,
        )

        assert result.shape == tensor.shape

        # not sure if we need to do this
        del tensor.grad
        del tensor

        return result

    def __call__(self, tensor: Tensor) -> Tensor:
        return self._reset_grad(self.update_weight(tensor))
