"""
Several layers can be grouped together into a single layer called a block
"""

from typing import Callable

import numpy as np

from tricycle.activation import GeLU
from tricycle.einsum import Einsum
from tricycle.functions import softmax
from tricycle.initialisers import init_xavier
from tricycle.layers import Dense, Dropout, Layer, LayerNorm
from tricycle.optimisers import Optimiser
from tricycle.tensor import Tensor, to_tensor


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
        attention_dropout_prob: float,
        residual_dropout_prob: float,
        initialiser=init_xavier,
    ):
        # set the constants
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads
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

        self.attention_dropout = Dropout(attention_dropout_prob)
        self.residual_dropout = Dropout(residual_dropout_prob)

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
        attention = self.attention_dropout(attention)

        # smush the heads back together
        out_shape = (n_tokens, self.embedding_dim)
        out = Einsum("NIj, NjH -> INH")(attention, value).reshape(out_shape)
        out = self.residual_dropout(out)
        return out

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


class MLPBlock(Layer):
    """
    A simple GPT-2 style MLP block with 2 linear layers around an activation
    function

    The size of the hidden dimension is expansion_ratio * the size of the
    input
    """

    embedding_dim: int
    dropout_prob: float
    expansion_ratio: float
    activation_fn: Callable
    linear_1: Dense
    linear_2: Dense

    def __init__(
        self,
        embedding_dim: int,
        dropout_prob: float,
        expansion_ratio: float = 4,
        activation_fn: Callable = GeLU(),
    ):
        self.linear_1 = Dense(
            from_size=embedding_dim,
            to_size=int(expansion_ratio * embedding_dim),
            initialiser=init_xavier,
        )
        self.linear_2 = Dense(
            from_size=int(expansion_ratio * embedding_dim),
            to_size=embedding_dim,
            initialiser=init_xavier,
        )
        self.dropout = Dropout(dropout_prob)
        self.activation_fn = activation_fn

    def forward(self, x: Tensor):
        x = self.linear_1(x)
        x = self.activation_fn(x)
        x = self.linear_2(x)
        x = self.dropout(x)
        return x

    def update(self, optimiser: Optimiser):
        self.linear_1.update(optimiser)
        self.linear_2.update(optimiser)

    def zero_grad(self):
        self.linear_1.zero_grad()
        self.linear_2.zero_grad()


class GPT2TransformerBlock(Layer):
    embedding_dim: int
    expansion_ratio: float
    activation_fn: Callable
    attention_dropout_prob: float
    residual_dropout_prob: float
    linear_dropout_prob: float

    def __init__(
        self,
        embedding_dim: int,
        n_heads: int,
        context_window: int,
        expansion_ratio: float = 4,
        activation_fn: Callable = GeLU(),
        attention_dropout_prob: float = 0,
        residual_dropout_prob: float = 0,
        linear_dropout_prob: float = 0,
    ):
        self.attention_block = MultiHeadSelfAttention(
            embedding_dim,
            n_heads=n_heads,
            context_window=context_window,
            attention_dropout_prob=attention_dropout_prob,
            residual_dropout_prob=residual_dropout_prob,
            initialiser=init_xavier,
        )
        self.mlp_block = MLPBlock(
            embedding_dim,
            linear_dropout_prob,
            expansion_ratio,
            activation_fn,
        )
        self.layer_norm_1 = LayerNorm()
        self.layer_norm_2 = LayerNorm()

    def forward(self, x: Tensor):
        x = self.attention_block(self.layer_norm_1(x)) + x
        x = self.mlp_block(self.layer_norm_2(x)) + x
        return x

    def update(self, optimiser: Optimiser):
        self.attention_block.update(optimiser)
        self.mlp_block.update(optimiser)

    def zero_grad(self):
        self.attention_block.zero_grad()
        self.mlp_block.zero_grad()