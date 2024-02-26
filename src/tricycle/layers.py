from abc import abstractmethod
from typing import Sequence

import numpy as np

from tricycle.binary import badd
from tricycle.initialisers import init_xavier
from tricycle.ops import einsum, reshape, softmax, split
from tricycle.optimisers import Optimiser
from tricycle.tensor import Tensor, to_tensor


class Layer:
    @abstractmethod
    def forward(self, x: Tensor):
        raise NotImplementedError

    def __call__(self, x: Tensor):
        return self.forward(x)

    @abstractmethod
    def update(self, optimiser: Optimiser):
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

    def update(self, optimiser: Optimiser):
        self.weights = optimiser(self.weights)

    def zero_grad(self):
        self.weights.grad = None

    def vectorise(self) -> "Dense":
        self._forward_op = einsum("za,ab->zb")
        return self


class MultiHeadSelfAttention(Layer):
    """
    Multi-head self-attention
    """

    embedding_dim: int
    n_heads: int
    dropout: float
    context_window: int

    def __init__(
        self,
        embedding_dim: int,
        n_heads: int,
        context_window: int,
        dropout: float,
        initialiser=init_xavier,
    ):
        # set the constants
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads
        self.dropout = dropout
        self.context_window = embedding_dim

        # Initialise the weights before and after the actual attention
        # mechanism. There aren't actually any weights in the attention bit
        # only before and after

        # Project the embedding into 3 embeddings. One for each of key, query
        # and value
        self.in_projection = Dense(
            in_features=self.embedding_dim,
            out_features=self.embedding_dim * 3,
            initialiser=initialiser,
        )

        # Pass the final embedding through a linear layer
        self.out_projection = Dense(
            in_features=self.embedding_dim,
            out_features=self.embedding_dim,
            initialiser=initialiser,
        )

        # build a mask to make attention causal
        self.mask = np.ones((context_window, context_window))
        idx = np.tril(self.mask.astype(bool))
        self.mask[~idx] = -np.inf
        self.mask[idx] = 0

    def _attention(self, key: Tensor, query: Tensor, value: Tensor):
        # reshape into n_heads x embedding_dim
        head_size = self.embedding_dim // self.n_heads
        n_tokens = key.shape[0]
        new_shape = (
            n_tokens,  # number of tokens
            self.n_heads,  # number of heads
            head_size,  # embedding per head
        )
        key = reshape(key, new_shape)
        query = reshape(query, new_shape)
        value = reshape(value, new_shape)

        # split into heads
        swap = einsum("tnh->nth")
        key = swap(key)
        query = swap(query)
        value = swap(value)

        # attend
        attend = einsum("nih,njh->nij")
        attention = attend(query, key) / np.sqrt(head_size)

        # mask
        mask = np.stack([self.mask[:n_tokens, :n_tokens]] * attention.shape[0])
        mask = to_tensor(mask, requires_grad=False, name="mask")

        # TODO: check if this breaks the gradients
        attention = badd(attention, mask)

        # softmax
        attention = softmax(attention)

        # smush the heads back together
        attention = einsum("nij,njh->nih")(attention, value)
        attention = swap(attention)
        return reshape(attention, (n_tokens, self.embedding_dim))

    def forward(self, x: Tensor):
        # use the projection layer to expand the inoput embedding
        x = self.in_projection(x)

        # split the embedding into key, query and value
        key, query, value = split(x, 3)

        attention = self._attention(key, query, value)

        # project back
        return self.out_projection(attention)

    def update(self, optimiser: Optimiser):
        self.weights = optimiser(self.weights)

    def zero_grad(self):
        self.weights.grad = None


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

    def vectorise(self) -> "Sequential":
        self.layers = [layer.vectorise() for layer in self.layers]
        return self
