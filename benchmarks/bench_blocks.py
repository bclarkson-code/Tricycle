import numpy as np

from tricycle.blocks import (
    GPT2TransformerBlock,
    GPT2TransformerBlockV2,
    GPT2TransformerBlockV3,
    MLPBlock,
    MLPBlock2,
    MLPBlock3,
    MLPBlock4,
    MultiHeadSelfAttention,
    MultiHeadSelfAttentionV2,
)
from tricycle.tensor import to_tensor

N_LOOPS = 10


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


def original_attention_block():
    batch_size = 16
    embedding_dim = 384
    n_heads = 6
    dropout = 0.2
    context_window = 128
    n_tokens = 256

    inputs = to_tensor(
        np.random.random(size=(batch_size, context_window, embedding_dim)),
        requires_grad=False,
    ).to_gpu()
    inputs = inputs.to_vector()
    block = MultiHeadSelfAttention(
        embedding_dim=embedding_dim,
        n_heads=n_heads,
        context_window=context_window,
        attention_dropout_prob=dropout,
        residual_dropout_prob=dropout,
    )
    block.to_gpu()

    for _ in range(N_LOOPS):
        out = block(inputs)
        out.backward()
        out.cleanup()


def new_attention_block():
    batch_size = 16
    embedding_dim = 384
    n_heads = 6
    dropout = 0.2
    context_window = 128

    inputs = to_tensor(
        np.random.random(size=(batch_size, context_window, embedding_dim)),
        requires_grad=False,
    ).to_gpu()
    inputs = inputs.to_vector()
    block = MultiHeadSelfAttentionV2(
        embedding_dim=embedding_dim,
        n_heads=n_heads,
        context_window=context_window,
        attention_dropout_prob=dropout,
        residual_dropout_prob=dropout,
    )
    block.to_gpu()

    for _ in range(N_LOOPS):
        out = block(inputs)
        out.backward()
        out.cleanup()


def original_attention_block_device_1():
    batch_size = 16
    embedding_dim = 384
    n_heads = 6
    dropout = 0.2
    context_window = 128

    inputs = to_tensor(
        np.random.random(size=(batch_size, context_window, embedding_dim)),
        requires_grad=False,
    ).to_gpu(1)
    inputs = inputs.to_vector()
    block = MultiHeadSelfAttention(
        embedding_dim=embedding_dim,
        n_heads=n_heads,
        context_window=context_window,
        attention_dropout_prob=dropout,
        residual_dropout_prob=dropout,
    )
    block.to_gpu(1)

    for _ in range(N_LOOPS):
        out = block(inputs)
        out.backward()
        out.cleanup()


def new_attention_block_device_1():
    batch_size = 16
    embedding_dim = 384
    n_heads = 6
    dropout = 0.2
    context_window = 128

    inputs = to_tensor(
        np.random.random(size=(batch_size, context_window, embedding_dim)),
        requires_grad=False,
    ).to_gpu(1)
    inputs = inputs.to_vector()
    block = MultiHeadSelfAttentionV2(
        embedding_dim=embedding_dim,
        n_heads=n_heads,
        context_window=context_window,
        attention_dropout_prob=dropout,
        residual_dropout_prob=dropout,
    )
    block.to_gpu(1)

    for _ in range(N_LOOPS):
        out = block(inputs)
        out.backward()
        out.cleanup()


def original_transformer_block():
    batch_size = 16
    embedding_dim = 384
    n_heads = 6
    dropout = 0.2
    context_window = 128

    inputs = to_tensor(
        np.random.random(size=(batch_size, context_window, embedding_dim)),
        requires_grad=False,
    ).to_gpu(1)
    inputs = inputs.to_vector()
    block = GPT2TransformerBlock(
        embedding_dim=embedding_dim,
        n_heads=n_heads,
        context_window=context_window,
        attention_dropout_prob=dropout,
        residual_dropout_prob=dropout,
    )
    block.to_gpu(1)

    for _ in range(N_LOOPS):
        out = block(inputs)
        out.backward()
        out.cleanup()


def new_transformer_block():
    batch_size = 16
    embedding_dim = 384
    n_heads = 6
    dropout = 0.2
    context_window = 128

    inputs = to_tensor(
        np.random.random(size=(batch_size, context_window, embedding_dim)),
        requires_grad=False,
    ).to_gpu(1)
    inputs = inputs.to_vector()
    block = GPT2TransformerBlockV2(
        embedding_dim=embedding_dim,
        n_heads=n_heads,
        context_window=context_window,
        attention_dropout_prob=dropout,
        residual_dropout_prob=dropout,
    )
    block.to_gpu(1)

    for _ in range(N_LOOPS):
        out = block(inputs)
        out.backward()
        out.cleanup()


def new_transformer_block_and_rms_norm():
    batch_size = 16
    embedding_dim = 384
    n_heads = 6
    dropout = 0.2
    context_window = 128

    inputs = to_tensor(
        np.random.random(size=(batch_size, context_window, embedding_dim)),
        requires_grad=False,
    ).to_gpu(1)
    inputs = inputs.to_vector()
    block = GPT2TransformerBlockV3(
        embedding_dim=embedding_dim,
        n_heads=n_heads,
        context_window=context_window,
        attention_dropout_prob=dropout,
        residual_dropout_prob=dropout,
    )
    block.to_gpu(1)

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
    # (
    #     original_attention_block,
    #     new_attention_block,
    #     "Using CUDA softmax",
    # ),
    # (
    #     original_attention_block,
    #     original_attention_block_device_1,
    #     "Using device 1",
    # ),
    # (
    #     original_attention_block,
    #     new_attention_block_device_1,
    #     "Using CUDA softmax, device 1 and improved dropout and dense",
    # ),
    # (
    #     original_transformer_block,
    #     new_transformer_block,
    #     "Optimised every layer",
    # ),
    # (
    #     original_transformer_block,
    #     new_transformer_block_and_rms_norm,
    #     "Optimised every layer + swapped layer norm for rms",
    # ),
]
