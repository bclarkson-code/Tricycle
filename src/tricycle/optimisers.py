from tricycle.tensor import Tensor


class Optimiser:
    def __call__(self, tensor: Tensor) -> Tensor:
        raise NotImplementedError


class StochasticGradientDescent:
    def __init__(self, learning_rate: float, weight_decay: float | None = None):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def _reset_grad(self, tensor: Tensor):
        tensor.grad = None
        tensor.args = None
        tensor.back_fn = None
        return tensor

    def update_weight(self, tensor: Tensor):
        assert tensor.grad is not None

        grad = self.learning_rate * tensor.grad
        if self.weight_decay is None:
            return tensor - grad

        wd = self.learning_rate * self.weight_decay * tensor
        return tensor - wd - grad

    def __call__(self, tensor: Tensor) -> Tensor:
        return self._reset_grad(self.update_weight(tensor))
