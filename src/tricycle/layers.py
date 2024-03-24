from abc import abstractmethod
from typing import Sequence

import numpy as np

from tricycle.einsum import Einsum
from tricycle.initialisers import init_xavier
from tricycle.ops import reshape, softmax
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


class Dense(Layer):
    weights: Tensor
    from_size: int
    to_size: int

    def __init__(self, from_size: int, to_size: int, initialiser=init_xavier):
        self.weights = initialiser((from_size, to_size), name="weights")
        self.from_size = from_size
        self.to_size = to_size

    def forward(self, x: Tensor):
        return Einsum("...a,ab->...b")(x, self.weights)

    def update(self, optimiser: Optimiser):
        self.weights = optimiser(self.weights)

    def zero_grad(self):
        self.weights.grad = None


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
        self.context_window = context_window

        # Project the embedding into 3 embeddings. One for each of key, query
        # and value
        self.in_projection = Dense(
            from_size=self.embedding_dim,
            to_size=self.embedding_dim * 3,
            initialiser=initialiser,
        )

        # Pass the final embedding through a linear layer
        self.out_projection = Dense(
            from_size=self.embedding_dim,
            to_size=self.embedding_dim,
            initialiser=initialiser,
        )

        # build a mask to make attention causal
        self.mask = build_mask(self.context_window)

    def _attention(self, key: Tensor, query: Tensor, value: Tensor):
        # reshape into n_heads x embedding_dim
        head_size = self.embedding_dim // self.n_heads
        n_tokens = key.shape[1] if key.is_vector else key.shape[0]
        head_shape = (
            n_tokens,  # number of tokens
            self.n_heads,  # number of heads
            head_size,  # embedding per head
        )
        out_shape = (n_tokens, self.embedding_dim)

        # reshape and reorder the heads
        key = key.reshape(head_shape).e("TNH -> NTH")
        query = query.reshape(head_shape).e("TNH -> NTH")
        value = value.reshape(head_shape).e("TNH -> NTH")

        # attend
        attention = Einsum("NIh, NJh -> NIJ")(query, key) / np.sqrt(head_size)

        # mask and softmax
        attention = masked_fill(attention, (n_tokens, n_tokens), self.mask)
        attention = softmax(attention)

        # smush the heads back together
        out_shape = (n_tokens, self.embedding_dim)
        return Einsum("NIj, NjH -> INH")(attention, value).reshape(out_shape)

    def forward(self, x: Tensor):
        # use the projection layer to expand the inoput embedding
        x = self.in_projection(x)

        # split the embedding into key, query and value
        query, key, value = x.split(3, axis=1)  # tricycle

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


def build_mask(context_window: int):
    """
    Build an attention mask to stop the model from being able to see
    future tokens
    """
    NEGATIVE_INFINITY = -np.inf
    mask = np.ones((context_window, context_window))
    idx = np.tril(mask.astype(bool))
    mask[~idx] = NEGATIVE_INFINITY
    mask[idx] = 0
    return to_tensor(mask, requires_grad=False, name="mask")


def masked_fill(x: Tensor, mask_shape: tuple[int, int], full_mask: Tensor):
    """
    Apply an attention_mask to a tensor
    """
    repeats = x.shape[1] if x.is_vector else x.shape[0]
    mask = np.stack([full_mask[: mask_shape[0], : mask_shape[1]]] * repeats)
    mask = to_tensor(mask, requires_grad=False, name="mask")
    return x + mask
