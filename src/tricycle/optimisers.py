import numpy as np

from tricycle.tensor import Tensor, to_tensor


class Optimiser:
    def __call__(self, tensor: Tensor) -> Tensor:
        raise NotImplementedError


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

    def _reset_grad(self, tensor: Tensor):
        tensor.grad = None
        tensor.args = None
        tensor.back_fn = None
        return tensor

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
            grad += wd

        if self.momentum is not None:
            if tensor.uuid not in self.momentum_store:
                last_momentum = to_tensor(np.zeros_like(grad))
            else:
                last_momentum = self.momentum_store[tensor.uuid]

            grad += self.momentum * last_momentum
            self.momentum_store[tensor.uuid] = grad

        del tensor.grad

        # We need to make sure that the new tensor looks like the old one
        old_uuid = tensor.uuid
        result = tensor - grad
        result.grad = None
        result.uuid = old_uuid

        return result

    def __call__(self, tensor: Tensor) -> Tensor:
        return self._reset_grad(self.update_weight(tensor))
