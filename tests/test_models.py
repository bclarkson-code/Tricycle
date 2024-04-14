import numpy as np

from tricycle.configs import GPTConfig
from tricycle.models import GPT, GPTV2
from tricycle.tensor import to_tensor


def test_gpt2():
    np.random.seed(0)
    config = GPTConfig()
    config.vocab_size = 17
    config.context_window = 13
    config.batch_size = 11
    config.n_tokens = 5
    config.n_heads = 3
    config.n_layers = 1
    config.embedding_dim = 7 * config.n_heads
    config.input_dropout_prob = 0.2
    config.attention_dropout_prob = 0.2
    config.expansion_ratio = 4
    config.activation_fn = "gelu"

    in_tensor = to_tensor(
        np.random.randint(
            0,
            config.vocab_size,
            (config.batch_size, config.n_tokens),
        ),
        dtype=np.int8,
        requires_grad=False,
    ).to_vector()
    in_tensor.to_gpu(1)

    model = GPT(config)
    model.to_gpu(1)
    out_tensor = model(in_tensor)


def test_gpt2_v2():
    np.random.seed(0)
    config = GPTConfig()
    config.vocab_size = 17
    config.context_window = 13
    config.batch_size = 11
    config.n_tokens = 5
    config.n_heads = 3
    config.n_layers = 1
    config.embedding_dim = 7 * config.n_heads
    config.input_dropout_prob = 0.2
    config.attention_dropout_prob = 0.2
    config.expansion_ratio = 4
    config.activation_fn = "gelu"

    in_tensor = to_tensor(
        np.random.randint(
            0,
            config.vocab_size,
            (config.batch_size, config.n_tokens),
        ),
        dtype=np.int8,
        requires_grad=False,
    ).to_vector()
    in_tensor.to_gpu(1)

    model = GPTV2(config)
    model.to_gpu(1)
    out_tensor = model(in_tensor)

    assert out_tensor.shape == (
        config.batch_size,
        config.n_tokens,
        config.vocab_size,
    )
    out_tensor = out_tensor.from_vector()
    out_tensor = out_tensor.mean().mean().mean()

    out_tensor.backward()
