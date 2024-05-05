class GPTConfig:
    embedding_dim: int
    context_window: int
    vocab_size: int
    n_heads: int
    n_layers: int
    expansion_ratio: float
    activation_fn: str

    input_dropout_prob: float
    attention_dropout_prob: float
    residual_dropout_prob: float
    linear_dropout_prob: float

    max_learning_rate: float
    min_learning_rate: float
    warmup_steps: int
    weight_decay: float
    momentum = float
    beta1: float
    beta2: float

    steps: int
    eval_interval: int
    batch_size: int
    gradient_accumulation_steps: int

    device_idx: int

    mlflow_tracking_uri: str
    mlflow_experiment_name: str


class SmolGPTConfig(GPTConfig):
    embedding_dim = 384
    context_window = 256
    vocab_size = 65
    n_heads = 6
    n_layers = 6
    expansion_ratio = 4
    activation_fn = "gelu"

    input_dropout_prob = 0
    attention_dropout_prob = 0
    residual_dropout_prob = 0
    linear_dropout_prob = 0

    max_learning_rate = 1e-3
    min_learning_rate = 1e-4
    warmup_steps = 100
    weight_decay = 1e-1
    momentum = 0
    beta1 = 0.9
    beta2 = 0.99

    steps = 5_000
    eval_interval = 50
    batch_size = 16
    gradient_accumulation_steps = 1

    device_idx = 0

    mlflow_enabled = True
    mlflow_tracking_uri = "http://localhost:5000"
