import textwrap

import numpy as np

from tricycle.layers import (
    Dense,
    DenseV2,
    Dropout,
    DropoutV2,
    DropoutV3,
    DropoutV4,
    DropoutV5,
    DropoutV6,
    Embedding,
    EmbeddingV2,
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
    layer = EmbeddingV2(from_size=vocab_size, to_size=embed_dim)

    for _ in range(10):
        out = layer(inputs)
        out.backward()
        out.zero_grad()


def test_embedding_original():
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


def test_dense_original():
    batch_size = 4
    shape = (256, 256)

    inputs = to_tensor(
        np.random.random(size=(batch_size, *shape)),
        requires_grad=True,
    )
    inputs = inputs.to_vector()
    layer = Dense(from_size=256, to_size=256)

    for _ in range(10):
        out = layer(inputs)
        out.backward()
        out.zero_grad()
        layer.zero_grad()


def test_dense_zero_grad_inputs():
    batch_size = 4
    shape = (256, 256)

    inputs = to_tensor(
        np.random.random(size=(batch_size, *shape)),
        requires_grad=False,
    )
    inputs = inputs.to_vector()
    layer = DenseV2(from_size=256, to_size=256)

    for _ in range(10):
        out = layer(inputs)
        out.backward()
        out.zero_grad()
        layer.zero_grad()

def test_dense_new_einsum():
    batch_size = 4
    shape = (256, 256)

    inputs = to_tensor(
        np.random.random(size=(batch_size, *shape)),
        requires_grad=False,
    )
    inputs = inputs.to_vector()
    layer = DenseV2(from_size=256, to_size=256)

    for _ in range(10):
        out = layer(inputs)
        out.backward()
        out.zero_grad()
        layer.zero_grad()

__benchmarks__ = [
    # (
    #     test_embedding_original,
    #     test_embedding_back_new,
    #     "Forward pass of embedding layer",
    # )
    #     (
    #         test_dropout_original,
    #         test_dropout_choice,
    #         "Use dropout with random.choice",
    #     ),
    #     (
    #         test_dropout_original,
    #         test_dropout_smaller_mask,
    #         "Use dropout with smaller mask",
    #     ),
    #     (
    #         test_dropout_original,
    #         test_dropout_bmask,
    #         "Use dropout with binary mask instead of mmul",
    #     ),
    #     (
    #         test_dropout_original,
    #         test_dropout_bmask_and_choice,
    #         textwrap.dedent(
    #             """Use dropout with binary mask instead of mmul
    #         and choice instead of binomial"""
    #         ),
    #     ),
    #     (
    #         test_dropout_original,
    #         test_dropout_bool_mask,
    #         "Use dropout with bool mask",
    #     ),
    (
        test_dense_original,
        test_dense_zero_grad_inputs,
        "Initial trial of dense layer",
    ),
]
