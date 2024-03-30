from typing import Callable

from tricycle.activation import GeLU


class GPTConfig:
    embedding_dim: int
    context_window: int
    vocab_size: int
    n_heads: int
    n_layers: int
    expansion_ratio: float
    activation_fn: Callable

    input_dropout_prob: float
    attention_dropout_prob: float
    residual_dropout_prob: float
    linear_dropout_prob: float


class PicoGPTConfig(GPTConfig):
    embedding_dim = 768
    context_window = 256
    vocab_size = 1024
    n_heads = 8
    n_layers = 1
    expansion_ratio = 4
    activation_fn = GeLU()

    input_dropout_prob = 0.1
    attention_dropout_prob = 0.1
    residual_dropout_prob = 0.1
    linear_dropout_prob = 0.1
