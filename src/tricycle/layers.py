from abc import ABC, abstractmethod
from string import ascii_letters
from typing import Sequence

import numpy as np

from tricycle.binary import bmul
from tricycle.einsum import Einsum
from tricycle.initialisers import init_xavier
from tricycle.optimisers import Optimiser
from tricycle.tensor import Tensor, nothing, to_tensor


class Layer(ABC):
    tensors: dict[str, Tensor] = {}
    layers: Sequence["Layer"] = []

    @abstractmethod
    def forward(self, tensor: Tensor):
        raise NotImplementedError

    def __call__(self, tensor: Tensor):
        return self.forward(tensor)

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
        self.tensors = {"weights": self.weights}

    def _build_missing_indices(
        self, tensor: Tensor, initial_subscript: str
    ) -> str:
        """
        In some circumstances, using ellipses with vectorised tensors
        can be defined in the forward direction but not in reverse.

        To fix this, we're building a string of indices that can be used
        in place of an ellipsis. This is a bit of an ugly hack, but it
        works for now.

        TODO: fix this properly
        """
        n_untouched_indices = (
            len(tensor.shape) - 2
            if tensor.is_vector
            else len(tensor.shape) - 1
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

    def forward(self, tensor: Tensor):
        initial_subscript = "a,aB->B"
        idx = self._build_missing_indices(tensor, initial_subscript)
        return Einsum(f"{idx}a,aB->{idx}B")(tensor, self.weights)

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

    def forward(self, tensor: Tensor):
        random_mask = np.random.binomial(
            n=1, p=1 - self.probability, size=tensor.shape
        )
        random_mask = to_tensor(
            random_mask, requires_grad=False, is_vector=tensor.is_vector
        )
        return bmul(tensor, random_mask)


class LayerNorm(Layer):
    """
    Normalise each tensor individually
    """

    def forward(self, tensor: Tensor):
        return tensor.normalise()


class Embedding(Layer):
    """
    Convert an index to an embedding with a lookup (rather than a one-hot
    encoding and a matrix multiplication)
    """

    def __init__(self, from_size: int, to_size: int, initialiser=init_xavier):
        self.weights = initialiser((from_size, to_size))
        self.vocab_size = from_size

    def forward(self, tensor: Tensor):
        assert (
            tensor.requires_grad is False
        ), "Cannot embed a differentiable tensor"

        if tensor.is_vector:
            result = tensor.xp.stack(
                [self.weights._data[idx] for idx in tensor._data]
            )
            result = to_tensor(
                result,
                is_vector=True,
            )
        else:
            result = to_tensor(self.weights[tensor._data], is_vector=False)

        result.args = (tensor, self.weights)

        def _embed_back_fn(grad: Tensor):
            xp = grad.xp
            coef = xp.zeros((tensor.shape[-1], self.vocab_size))
            indices = xp.arange(tensor.shape[-1])

            coef[indices, tensor._data] = 1
            coef = to_tensor(coef, requires_grad=False)
            return Einsum("aB,aC->BC")(coef, grad)

        result.back_fns = (nothing, _embed_back_fn)
        return result

    def _raise_exception(self, *args):
        """
        I haven't figured out how 2nd order derivatives work yet so we'll
        raise an exception for now
        """
        raise NotImplementedError(
            "2nd order derivatives for embedding are not yet implemented"
        )

    def update(self, optimiser: Optimiser):
        self.weights = optimiser(self.weights)

    def zero_grad(self):
        self.weights.grad = None

    def to_gpu(self):
        self.weights.to_gpu()

    def from_gpu(self):
        self.weights.from_gpu()


class Sequential(Layer):
    def __init__(self, *layers: Layer):
        self.layers = layers

    def __getitem__(self, idx):
        return self.layers[idx]

    def forward(self, tensor: Tensor):
        for layer in self.layers:
            tensor = layer(tensor)
        return tensor

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
