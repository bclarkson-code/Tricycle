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

    learning_rate: float
    weight_decay: float
    momentum: float

    batch_size: int

    mlflow_enabled: bool
    mlflow_tracking_uri: str
    mlflow_experiment_name: str


class SmolGPTConfig(GPTConfig):
    embedding_dim = 128
    context_window = 64
    vocab_size = 1024
    n_heads = 2
    n_layers = 1
    expansion_ratio = 4
    activation_fn = GeLU()

    input_dropout_prob = 0
    attention_dropout_prob = 0
    residual_dropout_prob = 0
    linear_dropout_prob = 0

    learning_rate = 3e-2
    weight_decay = 0
    momentum = 0

    batch_size = 32

    debug = True
    mlflow_enabled = True
    mlflow_tracking_uri = "http://localhost:5000"
    mlflow_experiment_name = "Tricycle SmolGPT"
