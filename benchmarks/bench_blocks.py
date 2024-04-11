import numpy as np

from tricycle.blocks import MLPBlock, MLPBlock2, MLPBlock3, MLPBlock4
from tricycle.tensor import to_tensor

N_LOOPS = 25


def original_MLP_block():
    batch_size = 12
    embedding_dim = 384
    n_tokens = 256

    inputs = to_tensor(
        np.random.random(size=(batch_size, n_tokens, embedding_dim)),
        requires_grad=False,
    ).to_gpu()
    inputs = inputs.to_vector()
    block = MLPBlock(embedding_dim=embedding_dim, dropout_prob=0.2).to_gpu()

    for _ in range(N_LOOPS):
        out = block(inputs)
        out.backward()
        out.cleanup()


def MLP_block_new_dropout_and_new_dense():
    batch_size = 12
    embedding_dim = 384
    n_tokens = 256

    inputs = to_tensor(
        np.random.random(size=(batch_size, n_tokens, embedding_dim)),
        requires_grad=True,
    ).to_gpu()
    inputs = inputs.to_vector()
    block = MLPBlock2(embedding_dim=embedding_dim, dropout_prob=0.2).to_gpu()

    for _ in range(N_LOOPS):
        out = block(inputs)
        out.backward()
        out.cleanup()


def MLP_block_new_dense_dropout_efficient_gelu():
    batch_size = 12
    embedding_dim = 384
    n_tokens = 256

    inputs = to_tensor(
        np.random.random(size=(batch_size, n_tokens, embedding_dim)),
        requires_grad=True,
    ).to_gpu()
    inputs = inputs.to_vector()
    block = MLPBlock3(embedding_dim=embedding_dim, dropout_prob=0.2).to_gpu()

    for _ in range(N_LOOPS):
        out = block(inputs)
        out.backward()
        out.cleanup()


def MLP_block_new_dense_dropout_relu():
    batch_size = 12
    embedding_dim = 384
    n_tokens = 256

    inputs = to_tensor(
        np.random.random(size=(batch_size, n_tokens, embedding_dim)),
        requires_grad=True,
    ).to_gpu()
    inputs = inputs.to_vector()
    block = MLPBlock4(embedding_dim=embedding_dim, dropout_prob=0.2).to_gpu()

    for _ in range(N_LOOPS):
        out = block(inputs)
        out.backward()
        out.cleanup()


__benchmarks__ = [
    # (
    #     original_MLP_block,
    #     MLP_block_new_dropout_and_new_dense,
    #     "Combining improvements in dropout and dense",
    # ),
    # (
    #     original_MLP_block,
    #     MLP_block_new_dense_dropout_efficient_gelu,
    #     "Combining improvements in dropout, dense, and gelu",
    # ),
    # (
    #     original_MLP_block,
    #     MLP_block_new_dense_dropout_relu,
    #     "Combining improvements in dropout, dense, and relu",
    # ),
]
