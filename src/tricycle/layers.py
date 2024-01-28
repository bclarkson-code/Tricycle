from abc import abstractmethod
from typing import Sequence

from tricycle.initialisers import init_xavier
from tricycle.ops import einsum
from tricycle.tensor import Tensor, to_tensor


class Layer:
    @abstractmethod
    def forward(self, x: Tensor):
        raise NotImplementedError

    def __call__(self, x: Tensor):
        return self.forward(x)

    @abstractmethod
    def update(self, learning_rate: float):
        raise NotImplementedError

    @abstractmethod
    def zero_grad(self):
        raise NotImplementedError

    @abstractmethod
    def vectorise(self) -> "Layer":
        raise NotImplementedError


class Dense(Layer):
    weights: Tensor
    in_features: int
    out_features: int
    _forward_op: einsum

    def __init__(self, in_features: int, out_features: int, initialiser=init_xavier):
        self.weights = initialiser((in_features, out_features), name="weights")
        self.in_features = in_features
        self.out_features = out_features
        self._forward_op = einsum("a,ab->b")

    def forward(self, x: Tensor):
        return self._forward_op(x, self.weights)

    def update(self, learning_rate: float):
        self.weights = self.weights - self.weights.grad * learning_rate
        self.weights.grad = None
        self.weights.args = None
        self.weights.back_fn = None

    def zero_grad(self):
        self.weights.grad = None

    def vectorise(self) -> "Dense":
        self._forward_op = einsum("za,ab->zb")
        return self


class Sequential(Layer):
    layers: Sequence[Layer]

    def __init__(self, *layers: Layer):
        self.layers = layers

    def __getitem__(self, idx):
        return self.layers[idx]

    def forward(self, x: Tensor):
        for layer in self.layers:
            x = layer(x)
        return x

    def update(self, learning_rate: float):
        for layer in self.layers:
            layer.update(learning_rate)

    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()

    def vectorise(self) -> "Sequential":
        self.layers = [layer.vectorise() for layer in self.layers]
        return self
