from tricycle.activation import GeLU
from tricycle.layers import Layer


class GPTConfig:
    embedding_dim: int
    context_window: int
    vocab_size: int
    n_heads: int
    n_layers: int
    expansion_ratio: float
    activation_fn: Layer

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

    pad_token_id: int


class SmolGPTConfig(GPTConfig):
    embedding_dim = 384
    context_window = 256
    vocab_size = 65
    n_heads = 6
    n_layers = 6
    expansion_ratio = 4
    activation_fn = GeLU()

    input_dropout_prob = 0
    attention_dropout_prob = 0
    residual_dropout_prob = 0
    linear_dropout_prob = 0

    max_learning_rate = 1e-3
    min_learning_rate = 1e-4
    warmup_steps = 100
    steps = 5_000
    weight_decay = 1e-1
    momentum = 0

    beta1 = 0.9
    beta2 = 0.99
    gradient_accumulation_steps = 1

    batch_size = 16

    debug = True
    mlflow_enabled = True
    mlflow_tracking_uri = "http://localhost:5000"
    mlflow_experiment_name = "Tricycle SmolGPT"

    pad_token_id = 0
