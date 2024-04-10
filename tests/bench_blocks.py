import numpy as np

from tricycle.blocks import MLPBlock
from tricycle.tensor import to_tensor


def original_MLP_block():
    batch_size = 4
    embedding_dim = 32
    n_tokens = 64

    inputs = to_tensor(
        np.random.random(size=(batch_size, n_tokens, embedding_dim)),
        requires_grad=True,
    )
    inputs = inputs.to_vector()
    block = MLPBlock(embedding_dim=embedding_dim, dropout_prob=0.2)

    for _ in range(100):
        out = block(inputs)
        out.backward()
        out.zero_grad()


def new_MLP_block():
    batch_size = 4
    embedding_dim = 32
    n_tokens = 64

    inputs = to_tensor(
        np.random.random(size=(batch_size, n_tokens, embedding_dim)),
        requires_grad=True,
    )
    inputs = inputs.to_vector()
    block = MLPBlock(embedding_dim=embedding_dim, dropout_prob=0.2)

    for _ in range(100):
        out = block(inputs)
        out.backward()
        block.zero_grad()
        out.zero_grad()
        inputs.zero_grad()


__benchmarks__ = [
    # (original_MLP_block, new_MLP_block, "Base trial to find good params")
]
