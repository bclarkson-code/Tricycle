import textwrap

import numpy as np

from tricycle.layers import (
    Dense,
    DenseV2,
    DenseV3,
    DenseV4,
    Dropout,
    DropoutV2,
    DropoutV3,
    DropoutV4,
    DropoutV5,
    DropoutV6,
    DropoutV7,
    Embedding,
    LayerNorm,
    RMSNorm,
    RMSNormV2,
)
from tricycle.tensor import to_tensor


def test_embedding_back_new():
    n_tokens = 100
    vocab_size = 1024
    batch_size = 4
    embed_dim = 384

    inputs = to_tensor(
        np.random.randint(0, vocab_size, (batch_size, n_tokens)),
        requires_grad=False,
        dtype=int,
    )
    inputs = inputs.to_vector()
    layer = Embedding(from_size=vocab_size, to_size=embed_dim)

    for _ in range(10):
        out = layer(inputs)
        out.backward()
        out.zero_grad()


def test_dropout_original():
    batch_size = 4
    shape = (256, 256)

    inputs = to_tensor(
        np.random.random(size=(batch_size, *shape)),
        requires_grad=True,
    )
    inputs = inputs.to_vector()
    layer = Dropout(probability=0.2)

    for _ in range(10):
        out = layer(inputs)
        out.backward()
        out.zero_grad()


def test_dropout_choice():
    batch_size = 4
    shape = (256, 256)

    inputs = to_tensor(
        np.random.random(size=(batch_size, *shape)),
        requires_grad=True,
    )
    inputs = inputs.to_vector()
    layer = DropoutV2(probability=0.2)

    for _ in range(10):
        out = layer(inputs)
        out.backward()
        out.zero_grad()


def test_dropout_bmask():
    batch_size = 4
    shape = (256, 256)

    inputs = to_tensor(
        np.random.random(size=(batch_size, *shape)),
        requires_grad=True,
    )
    inputs = inputs.to_vector()
    layer = DropoutV3(probability=0.2)

    for _ in range(10):
        out = layer(inputs)
        out.backward()
        out.zero_grad()


def test_dropout_bmask_and_choice():
    batch_size = 4
    shape = (256, 256)

    inputs = to_tensor(
        np.random.random(size=(batch_size, *shape)),
        requires_grad=True,
    )
    inputs = inputs.to_vector()
    layer = DropoutV4(probability=0.2)

    for _ in range(10):
        out = layer(inputs)
        out.backward()
        out.zero_grad()


def test_dropout_smaller_mask():
    batch_size = 4
    shape = (256, 256)

    inputs = to_tensor(
        np.random.random(size=(batch_size, *shape)),
        requires_grad=True,
    )
    inputs = inputs.to_vector()
    layer = DropoutV5(probability=0.2)

    for _ in range(10):
        out = layer(inputs)
        out.backward()
        out.zero_grad()


def test_dropout_bool_mask():
    batch_size = 4
    shape = (256, 256)

    inputs = to_tensor(
        np.random.random(size=(batch_size, *shape)),
        requires_grad=True,
    )
    inputs = inputs.to_vector()
    layer = DropoutV6(probability=0.2)

    for _ in range(10):
        out = layer(inputs)
        out.backward()
        out.zero_grad()


def test_dropout_small_shape_and_bool_mask():
    batch_size = 4
    shape = (256, 256)

    inputs = to_tensor(
        np.random.random(size=(batch_size, *shape)),
        requires_grad=True,
    )
    inputs = inputs.to_vector()
    layer = DropoutV7(probability=0.2)

    for _ in range(10):
        out = layer(inputs)
        out.backward()
        out.zero_grad()


def test_dense_original():
    batch_size = 12
    shape = (1024, 384)
    to_shape = 384 * 4

    inputs = to_tensor(
        np.random.random(size=(batch_size, *shape)),
        requires_grad=True,
    ).to_gpu(1)
    inputs = inputs.to_vector()
    layer = Dense(from_size=shape[1], to_size=to_shape).to_gpu(1)

    for _ in range(100):
        out = layer(inputs)
        out.backward()
        out.cleanup()
        out.zero_grad()
        layer.zero_grad()


def test_dense_zero_grad_inputs():
    batch_size = 12
    shape = (1024, 384)
    to_shape = 384 * 4

    inputs = to_tensor(
        np.random.random(size=(batch_size, *shape)),
        requires_grad=False,
    ).to_gpu(1)
    inputs = inputs.to_vector()
    layer = DenseV2(from_size=shape[1], to_size=to_shape).to_gpu(1)

    for _ in range(100):
        out = layer(inputs)
        out.backward()
        out.cleanup()
        out.zero_grad()
        layer.zero_grad()


def test_dense_hand_crafted_derivative():
    batch_size = 12
    shape = (1024, 384)
    to_shape = 384 * 4

    inputs = to_tensor(
        np.random.random(size=(batch_size, *shape)),
        requires_grad=True,
    ).to_gpu(1)
    inputs = inputs.to_vector()
    layer = DenseV3(from_size=shape[1], to_size=to_shape).to_gpu(1)

    for _ in range(100):
        out = layer(inputs)
        out.backward()
        out.cleanup()


def test_dense_no_einsum():
    batch_size = 12
    shape = (1024, 384)
    to_shape = 384 * 4

    inputs = to_tensor(
        np.random.random(size=(batch_size, *shape)),
        requires_grad=True,
    ).to_gpu(1)
    inputs = inputs.to_vector()
    layer = DenseV4(from_size=shape[1], to_size=to_shape).to_gpu(1)

    for _ in range(100):
        out = layer(inputs)
        out.backward()
        out.cleanup()


def layer_norm_original():
    batch_size = 12
    shape = (1024, 384)

    inputs = to_tensor(
        np.random.random(size=(batch_size, *shape)),
        requires_grad=True,
    ).to_gpu(1)
    inputs = inputs.to_vector()
    layer = LayerNorm()

    for _ in range(10):
        out = layer(inputs)
        out.backward()
        out.cleanup()


def rms_norm_original():
    batch_size = 12
    shape = (1024, 384)

    inputs = to_tensor(
        np.random.random(size=(batch_size, *shape)),
        requires_grad=True,
    ).to_gpu(1)
    inputs = inputs.to_vector()
    layer = RMSNorm()

    for _ in range(10):
        out = layer(inputs)
        out.backward()
        out.cleanup()


def rms_norm_hand_crafted_derivative():
    batch_size = 12
    shape = (1024, 384)

    inputs = to_tensor(
        np.random.random(size=(batch_size, *shape)),
        requires_grad=True,
    ).to_gpu(1)
    inputs = inputs.to_vector()
    layer = RMSNormV2()

    for _ in range(10):
        out = layer(inputs)
        out.backward()
        out.cleanup()


__benchmarks__ = [
    # (
    #     test_embedding_original,
    #     test_embedding_back_new,
    #     "Forward pass of embedding layer",
    # )
    # (
    #     test_dropout_original,
    #     test_dropout_choice,
    #     "Use dropout with random.choice",
    # ),
    # (
    #     test_dropout_original,
    #     test_dropout_smaller_mask,
    #     "Use dropout with smaller mask",
    # ),
    # (
    #     test_dropout_original,
    #     test_dropout_bmask,
    #     "Use dropout with binary mask instead of mmul",
    # ),
    # (
    #     test_dropout_original,
    #     test_dropout_bmask_and_choice,
    #     textwrap.dedent(
    #         """Use dropout with binary mask instead of mmul
    #         and choice instead of binomial"""
    #     ),
    # ),
    # (
    #     test_dropout_original,
    #     test_dropout_bool_mask,
    #     "Use dropout with bool mask",
    # ),
    # (
    #     test_dropout_original,
    #     test_dropout_small_shape_and_bool_mask,
    #     "Use dropout with small shape and bool mask",
    # ),
    # (
    #     test_dense_original,
    #     test_dense_zero_grad_inputs,
    #     "Not calculating derivative for inputs",
    # ),
    # (
    #     test_dense_original,
    #     test_dense_hand_crafted_derivative,
    #     "Hand crafting derivatives",
    # ),
    # (
    #     test_dense_original,
    #     test_dense_no_einsum,
    #     "No Einsum",
    # ),
    # (
    #     layer_norm_original,
    #     rms_norm_original,
    #     "swapped layer norm for rms norm",
    # ),
    # (
    #     rms_norm_original,
    #     rms_norm_hand_crafted_derivative,
    #     "Hand crafted rms norm",
    # ),
]
