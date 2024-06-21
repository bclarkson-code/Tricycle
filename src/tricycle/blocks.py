"""
Several layers can be grouped together into a single layer called a block
"""

from typing import Literal

import numpy as np

from tricycle.activation import GLU, GeLU, ReLU, SwiGLU, Swish
from tricycle.attention import Attention
from tricycle.initialisers import init_xavier
from tricycle.layers import (  # noqa E501
    Dense,
    Dropout,
    Layer,
    LayerNorm,
    RMSNorm,
)
from tricycle.optimisers import Optimiser
from tricycle.tensor import Tensor, to_tensor


def build_mask(context_window: int) -> Tensor:
    """
    Build an attention mask to stop the model from being able to see
    future tokens
    """
    NEGATIVE_INFINITY = -10_000
    mask = np.ones((context_window, context_window))
    idx = np.tril(mask.astype(bool))
    mask[~idx] = NEGATIVE_INFINITY
    mask[idx] = 0
    return to_tensor(mask, requires_grad=False, name="mask")


def masked_fill(
    tensor: Tensor, mask_shape: tuple[int, int], full_mask: Tensor
):
    """
    Apply an attention_mask to a tensor
    """
    xp = tensor.xp
    repeats = tensor.shape[1] if tensor.is_batched else tensor.shape[0]
    mask = xp.stack(
        [full_mask[: mask_shape[0], : mask_shape[1]].array] * repeats
    )
    mask = to_tensor(mask, requires_grad=False, name="mask")
    result = tensor + mask
    result.name = "masked"
    return result


class MultiHeadSelfAttention(Layer):
    """
    Multi-head self-attention
    """

    embedding_dim: int
    n_heads: int
    context_window: int

    def __init__(
        self,
        embedding_dim: int,
        n_heads: int,
        context_window: int,
        residual_dropout_prob: float = 0.0,
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
            name="in_projection",
        )

        # Pass the final embedding through a linear layer
        self.out_projection = Dense(
            from_size=self.embedding_dim,
            to_size=self.embedding_dim,
            initialiser=initialiser,
            name="out_projection",
        )

        self.residual_dropout = Dropout(residual_dropout_prob)
        self.layers = [
            self.in_projection,
            self.residual_dropout,
            self.out_projection,
        ]

        self.attention = Attention(
            embedding_dim=embedding_dim,
            n_heads=n_heads,
            context_window=context_window,
        )

    def forward(self, tensor: Tensor):
        # expand the input
        tensor = self.in_projection(tensor)
        attention = self.attention(tensor)

        # project back
        projected = self.out_projection(attention)
        projected = self.residual_dropout(projected)

        return projected

    def update(self, optimiser: Optimiser):
        self.in_projection.update(optimiser)
        self.out_projection.update(optimiser)

    def zero_grad(self):
        self.in_projection.zero_grad()
        self.out_projection.zero_grad()

    def to_gpu(self, device: int = 0):
        self.in_projection.to_gpu(device)
        self.out_projection.to_gpu(device)
        self.attention.to_gpu(device)

    def from_gpu(self):
        self.in_projection.from_gpu()
        self.out_projection.from_gpu()
        self.attention.from_gpu()


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
    activation_fn: Layer
    linear_1: Dense
    linear_2: Dense

    def __init__(
        self,
        embedding_dim: int,
        dropout_prob: float,
        expansion_ratio: float = 4,
        activation_fn: Layer | str = GeLU(),
    ):
        self.linear_1 = Dense(
            from_size=embedding_dim,
            to_size=int(expansion_ratio * embedding_dim),
            initialiser=init_xavier,
            name="linear_1",
        )
        self.linear_2 = Dense(
            from_size=int(expansion_ratio * embedding_dim),
            to_size=embedding_dim,
            initialiser=init_xavier,
            name="linear_2",
        )
        self.dropout = Dropout(dropout_prob)
        if isinstance(activation_fn, str):
            match activation_fn:
                case "gelu":
                    activation_fn = GeLU()
                case "relu":
                    activation_fn = ReLU()
                case "swish":
                    activation_fn = Swish()
                case "glu":
                    activation_fn = GLU(int(expansion_ratio * embedding_dim))
                case "swiglu":
                    activation_fn = SwiGLU(
                        int(expansion_ratio * embedding_dim)
                    )
                case _:
                    raise NotImplementedError(
                        f"Activation function {activation_fn} is not "
                        "yet implemented"
                    )
        self.activation_fn = activation_fn
        self.layers = [
            self.linear_1,
            self.activation_fn,
            self.linear_2,
            self.dropout,
        ]

    def forward(self, x: Tensor):
        x = self.linear_1(x)
        x = self.activation_fn(x)
        x = self.linear_2(x)
        x = self.dropout(x)
        return x

    def update(self, optimiser: Optimiser):
        self.linear_1.update(optimiser)
        self.linear_2.update(optimiser)
        return self

    def zero_grad(self):
        self.linear_1.zero_grad()
        self.linear_2.zero_grad()
        return self

    def to_gpu(self, device: int = 0):
        self.linear_1.to_gpu(device)
        self.linear_2.to_gpu(device)
        return self

    def from_gpu(self):
        self.linear_1.from_gpu()
        self.linear_2.from_gpu()
        return self


class GPT2TransformerBlock(Layer):
    embedding_dim: int
    expansion_ratio: float
    activation_fn: Layer
    residual_dropout_prob: float
    linear_dropout_prob: float

    def __init__(
        self,
        embedding_dim: int,
        n_heads: int,
        context_window: int,
        expansion_ratio: float = 4,
        activation_fn: Layer | str = GeLU(),
        norm_fn: Literal["layer_norm"] | Literal["rms_norm"] = "layer_norm",
        residual_dropout_prob: float = 0,
        linear_dropout_prob: float = 0,
    ):
        self.attention_block = MultiHeadSelfAttention(
            embedding_dim,
            n_heads=n_heads,
            context_window=context_window,
            residual_dropout_prob=residual_dropout_prob,
            initialiser=init_xavier,
        )
        self.mlp_block = MLPBlock(
            embedding_dim,
            linear_dropout_prob,
            expansion_ratio,
            activation_fn,
        )

        match norm_fn:
            case "layer_norm":
                self.norm_1 = LayerNorm(embedding_dim)
                self.norm_2 = LayerNorm(embedding_dim)
            case "rms_norm":
                self.norm_1 = RMSNorm(embedding_dim)
                self.norm_2 = RMSNorm(embedding_dim)
            case _:
                raise ValueError(f"Unknown norm: {norm_fn}")

        self.layers = [
            self.norm_1,
            self.attention_block,
            self.norm_2,
            self.mlp_block,
        ]

    def forward(self, x: Tensor):
        normed = self.norm_1(x)

        attn = self.attention_block(normed)
        attn += x

        x = self.norm_2(attn)

        x = self.mlp_block(x)
        x += attn

        return x

    def update(self, optimiser: Optimiser):
        self.attention_block.update(optimiser)
        self.mlp_block.update(optimiser)
        self.norm_1.update(optimiser)
        self.norm_2.update(optimiser)

    def zero_grad(self):
        self.attention_block.zero_grad()
        self.mlp_block.zero_grad()
        self.norm_1.zero_grad()
        self.norm_2.zero_grad()

    def to_gpu(self, device: int = 0):
        self.attention_block.to_gpu(device)
        self.mlp_block.to_gpu(device)
        self.norm_1.to_gpu(device)
        self.norm_2.to_gpu(device)

    def from_gpu(self):
        self.attention_block.from_gpu()
        self.mlp_block.from_gpu()
        self.norm_1.from_gpu()
        self.norm_1.from_gpu()
