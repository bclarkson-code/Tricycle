import numpy as np

from tricycle.blocks import GPT2TransformerBlock, MLPBlock
from tricycle.tensor import Tensor


def test_MLPBlock():
    np.random.seed(0)
    in_tensor = Tensor(np.arange(12, dtype=float).reshape(3, 4))
    block = MLPBlock(embedding_dim=4, expansion_ratio=4, dropout_prob=0.5)

    assert block.linear_1.weights.shape == (4, 16)
    assert block.linear_2.weights.shape == (16, 4)

    block.linear_1.weights = Tensor(np.ones(block.linear_1.weights.shape))
    block.linear_2.weights = Tensor(np.ones(block.linear_2.weights.shape))

    out_tensor = block(in_tensor.to_batched())

    assert out_tensor.shape == (3, 4)

    correct_output = np.array(
        [
            [192.0, 0.0, 192.0, 0.0],
            [
                0.0,
                0.0,
                704.0,
                704.0,
            ],
            [1216.0, 1216.0, 1216.0, 0.0],
        ]
    )
    correct_output = Tensor(correct_output)

    assert out_tensor.is_batched
    assert out_tensor.close_to(correct_output)

    out_tensor.backward()

    assert in_tensor.grad is not None
    correct_grad = Tensor(
        [
            [64.0, 64.0, 64.0, 64.0],
            [64.0, 64.0, 64.0, 64.0],
            [96.0, 96.0, 96.0, 96.0],
        ]
    )

    assert in_tensor.grad.close_to(correct_grad)


def test_GPT2TransformerBlock():
    np.random.seed(0)
    batch_size = 11
    n_tokens = 32
    n_heads = 3
    embedding_dim = 7 * n_heads

    in_tensor = Tensor(
        np.random.random((batch_size, n_tokens, embedding_dim)),
        is_batched=True,
    )
    block = GPT2TransformerBlock(
        embedding_dim=embedding_dim,
        n_heads=3,
        expansion_ratio=4,
        context_window=32,
    )

    out_tensor = block(in_tensor.to_batched())

    assert out_tensor.shape == (batch_size, n_tokens, embedding_dim)

    out_tensor.backward()

    assert in_tensor.grad is not None
    assert in_tensor.grad.shape == in_tensor.shape
