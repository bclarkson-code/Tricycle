from abc import abstractmethod
from string import ascii_letters
from typing import Sequence

import numpy as np

from tricycle.binary import bmul
from tricycle.einsum import Einsum
from tricycle.functions import softmax
from tricycle.initialisers import init_xavier
from tricycle.optimisers import Optimiser
from tricycle.tensor import Tensor, to_tensor


class Layer:
    @abstractmethod
    def forward(self, x: Tensor):
        raise NotImplementedError

    def __call__(self, x: Tensor):
        return self.forward(x)

    def update(self, optimiser: Optimiser):
        pass

    def zero_grad(self):
        pass

    def to_gpu(self):
        pass

    def from_gpu(self):
        pass


class Dense(Layer):
    weights: Tensor
    from_size: int
    to_size: int
    name: str | None

    def __init__(
        self, from_size: int, to_size: int, initialiser=init_xavier, name=None
    ):
        self.weights = initialiser(
            (from_size, to_size), name="weights" if name is None else name
        )
        self.from_size = from_size
        self.to_size = to_size

    def _build_missing_indices(self, x: Tensor, initial_subscript: str) -> str:
        """
        In some circumstances, using ellipses with vectorised tensors
        can be defined in the forward direction but not in reverse.

        To fix this, we're building a string of indices that can be used
        in place of an ellipsis. This is a bit of an ugly hack, but it
        works for now.

        TODO: fix this properly
        """
        n_untouched_indices = (
            len(x.shape) - 2 if x.is_vector else len(x.shape) - 1
        )
        untouched_indices = ""
        i = 0
        while len(untouched_indices) < n_untouched_indices:
            next_idx = ascii_letters[i]
            if (
                next_idx not in untouched_indices
                and next_idx != "z"
                and next_idx not in initial_subscript
            ):
                untouched_indices += next_idx
            i += 1
        return untouched_indices

    def forward(self, x: Tensor):
        initial_subscript = "a,aB->B"
        idx = self._build_missing_indices(x, initial_subscript)
        return Einsum(f"{idx}a,aB->{idx}B")(x, self.weights)

    def update(self, optimiser: Optimiser):
        self.weights = optimiser(self.weights)

    def zero_grad(self):
        self.weights.grad = None

    def to_gpu(self):
        self.weights.to_gpu()

    def from_gpu(self):
        self.weights.from_gpu()


class Dropout(Layer):
    def __init__(self, probability: float):
        self.probability = probability

    def forward(self, x: Tensor):
        random_mask = np.random.binomial(
            n=1, p=1 - self.probability, size=x.shape
        )
        random_mask = to_tensor(
            random_mask, requires_grad=False, is_vector=x.is_vector
        )
        return bmul(x, random_mask)


class LayerNorm(Layer):
    """
    Normalise each tensor individually
    """

    def forward(self, x: Tensor):
        return x.normalise()


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

    def update(self, optimiser: Optimiser):
        for layer in self.layers:
            layer.update(optimiser)

    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()

    def to_gpu(self):
        for layer in self.layers:
            layer.to_gpu()

    def from_gpu(self):
        for layer in self.layers:
            layer.from_gpu()
