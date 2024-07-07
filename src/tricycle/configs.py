from dataclasses import asdict, dataclass
from typing import Literal


class GPTConfig:
    """
    Base config for GPT models
    """

    embedding_dim: int
    context_window: int
    vocab_size: int
    n_heads: int
    n_layers: int
    expansion_ratio: float
    activation_fn: str
    norm_fn: str

    input_dropout_prob: float
    residual_dropout_prob: float
    linear_dropout_prob: float

    max_learning_rate: float
    min_learning_rate: float
    warmup_steps: int
    weight_decay: float
    momentum = float
    beta1: float
    beta2: float

    steps: int | Literal["chinchilla_optimal"]
    eval_interval: int
    batch_size: int
    gradient_accumulation_steps: int

    device_idx: int

    mlflow_tracking_uri: str
    mlflow_experiment_name: str

    def dict(self) -> dict[str, int | float | str | bool]:
        out = {}
        for k, v in self.__class__.__dict__.items():
            if k.startswith("__"):
                continue

            if callable(v):
                continue
            out[k] = v
        return out


class ShakespeareConfig(GPTConfig):
    embedding_dim = 384
    context_window = 256
    vocab_size = 1024
    n_heads = 6
    n_layers = 6
    expansion_ratio = 4
    activation_fn = "gelu"
    norm_fn = "layer_norm"

    input_dropout_prob = 0.2
    residual_dropout_prob = 0.2
    linear_dropout_prob = 0.2

    max_learning_rate = 1e-3
    min_learning_rate = 1e-4
    warmup_steps = 100
    weight_decay = 1e-1
    momentum = 0
    beta1 = 0.9
    beta2 = 0.99

    steps = 5000
    eval_interval = 250
    eval_steps = 128
    batch_size = 32
    gradient_accumulation_steps = 1
    sample_size = 512

    device_idx = 1

    mlflow_enabled = True
    mlflow_tracking_uri = "http://localhost:5000"


class SmolGPTConfig(GPTConfig):
    embedding_dim = 768
    context_window = 1024
    vocab_size = 50256
    n_heads = 12
    n_layers = 12
    expansion_ratio = 4
    activation_fn = "gelu"
    norm_fn = "rms_norm"

    input_dropout_prob = 0.2
    residual_dropout_prob = 0.2
    linear_dropout_prob = 0.2

    max_learning_rate = 6e-4
    min_learning_rate = 0
    warmup_steps = 150  # roughly matches andrej's warmup steps in llm.c
    weight_decay = 1e-1
    momentum = 0
    beta1 = 0.9
    beta2 = 0.95

    steps = "chinchilla_optimal"
    eval_interval = 100
    eval_steps = 128
    batch_size = 4
    gradient_accumulation_steps = 128  # effective batch size of 524288 tokens
    n_tokens_to_generate = 512

    device_idx = 0

    mlflow_enabled = True
    mlflow_tracking_uri = "http://localhost:5000"
