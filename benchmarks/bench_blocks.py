import numpy as np

from tricycle.blocks import MLPBlock, MLPBlock2
from tricycle.tensor import to_tensor


def original_MLP_block():
    batch_size = 4
    embedding_dim = 128
    n_tokens = 128

    inputs = to_tensor(
        np.random.random(size=(batch_size, n_tokens, embedding_dim)),
        requires_grad=False,
    )
    inputs = inputs.to_vector()
    block = MLPBlock(embedding_dim=embedding_dim, dropout_prob=0.2)

    for _ in range(10):
        out = block(inputs)
        out.backward()
        out = out.zero_grad()


def MLP_block_new_dropout_and_new_dense():
    batch_size = 4
    embedding_dim = 128
    n_tokens = 128

    inputs = to_tensor(
        np.random.random(size=(batch_size, n_tokens, embedding_dim)),
        requires_grad=True,
    )
    inputs = inputs.to_vector()
    block = MLPBlock2(embedding_dim=embedding_dim, dropout_prob=0.2)

    for _ in range(10):
        out = block(inputs)
        out.backward()
        block.zero_grad()
        out.zero_grad()
        inputs.zero_grad()


__benchmarks__ = [
    (
        original_MLP_block,
        MLP_block_new_dropout_and_new_dense,
        "Combining improvements in dropout and dense",
    )
]
