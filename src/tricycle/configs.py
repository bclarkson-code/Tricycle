from dataclasses import asdict, dataclass


class GPTConfig:
    embedding_dim: int
    context_window: int
    vocab_size: int
    n_heads: int
    n_layers: int
    expansion_ratio: float
    activation_fn: str

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

    steps: int | None
    eval_interval: int
    batch_size: int
    gradient_accumulation_steps: int

    device_idx: int

    mlflow_tracking_uri: str
    mlflow_experiment_name: str


class SmolGPTConfig(GPTConfig):
    embedding_dim = 384
    context_window = 256
    vocab_size = 100276
    n_heads = 6
    n_layers = 6
    expansion_ratio = 4
    activation_fn = "gelu"

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

    steps = None
    eval_interval = 250
    batch_size = 64
    gradient_accumulation_steps = 1

    device_idx = 1

    mlflow_enabled = True
    mlflow_tracking_uri = "http://localhost:5000"

    def dict(self) -> dict[str, int | float | str | bool]:
        out = {}
        for k, v in SmolGPTConfig.__dict__.items():
            if k.startswith("__"):
                continue

            if callable(v):
                continue
            out[k] = v
        return out
