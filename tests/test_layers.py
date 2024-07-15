from copy import copy

import numpy as np

from tricycle.layers import (  # noqa: E501
    Dense,
    Dropout,
    Embedding,
    LayerNorm,
    Sequential,
)
from tricycle.tensor import Tensor


def test_dense_layer():
    layer = Dense(10, 8)

    assert layer.weights.shape == (10, 8)

    x_in = Tensor(np.ones(10))

    x_out = layer(x_in)
    assert x_out.shape == (8,)


def test_sequential_layer():
    layer1 = Dense(10, 8)
    layer2 = Dense(8, 4)

    model = Sequential(layer1, layer2)

    assert model.layers[0].weights.shape == (10, 8)
    assert model.layers[1].weights.shape == (8, 4)

    x_in = Tensor(np.ones(10))

    x_out = model(x_in)
    assert x_out.shape == (4,)


def test_dropout():  # sourcery skip: square-identity
    np.random.seed(0)
    size = 100
    dropout_prob = 0.3

    # non-batched
    in_tensor = Tensor(np.random.normal(size=(size, size)), name="in_tensor")
    dropout = Dropout(dropout_prob)

    out_tensor = dropout(in_tensor.to_batched())

    assert out_tensor.shape == in_tensor.shape
    zero_x_idx, zero_y_idx = np.where(out_tensor.array == 0)
    n_zeros = len(zero_x_idx)
    expected_n_zeros = int(size * size * dropout_prob)

    assert n_zeros / size**2 - expected_n_zeros / size**2 < 0.05

    out_tensor.backward()

    assert in_tensor.grad is not None
    assert in_tensor.grad.shape == in_tensor.shape

    coef = 1 / (1 - dropout_prob)
    correct_grad = np.full(in_tensor.shape, coef)
    correct_grad[zero_x_idx, zero_y_idx] = 0

    assert in_tensor.grad.close_to(correct_grad)


def test_layer_norm():
    np.random.seed(0)
    in_tensor = Tensor(np.random.normal(size=(100, 100)), name="in_tensor")
    layer_norm = LayerNorm(100)
    out_tensor = layer_norm(in_tensor.to_batched())

    assert out_tensor.shape == in_tensor.shape
    out_tensor.backward()

    assert copy(out_tensor).mean().close_to(0, atol=1e-3)
    assert np.allclose(np.std(out_tensor.array), [1] * 100, atol=1e-7)

    assert in_tensor.grad is not None
    assert in_tensor.grad.shape == in_tensor.shape

    # not sure if this is correct. TODO: check
    assert in_tensor.grad.close_to(np.zeros(in_tensor.shape), atol=1e-3)


def test_embedding():
    np.random.seed(0)
    vocab_size = 3
    out_shape = 5
    in_tensor = Tensor(
        [0, 1, 2, 0],
        requires_grad=False,
        dtype=int,
    )

    embedding_layer = Embedding(from_size=vocab_size, to_size=out_shape)
    weights = np.indices((vocab_size * out_shape,)).reshape(
        vocab_size, out_shape
    )
    embedding_layer.weights = Tensor(weights)

    result = embedding_layer(in_tensor)

    assert result.shape == (4, 5)
    assert result[0].close_to(embedding_layer.weights[0])
    assert result[1].close_to(embedding_layer.weights[1])
    assert result[2].close_to(embedding_layer.weights[2])
    assert result[3].close_to(embedding_layer.weights[0])

    result.backward()

    assert embedding_layer.weights.grad is not None
    assert embedding_layer.weights.grad.shape == embedding_layer.weights.shape
    assert embedding_layer.weights.grad.close_to(
        [[2, 2, 2, 2, 2], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]
    )


def test_embedding_batched():
    np.random.seed(0)
    vocab_size = 3
    out_shape = 5
    in_tensor = Tensor(
        [[0, 1, 2, 0], [1, 2, 2, 1]],
        requires_grad=False,
        dtype=np.int8,
    ).to_batched()

    embedding_layer = Embedding(from_size=vocab_size, to_size=out_shape)
    weights = np.indices((vocab_size * out_shape,)).reshape(
        vocab_size, out_shape
    )
    embedding_layer.weights = Tensor(weights)

    result = embedding_layer(in_tensor)

    assert result.shape == (2, 4, 5)
    assert result[0][0].close_to(embedding_layer.weights[0])
    assert result[0][1].close_to(embedding_layer.weights[1])
    assert result[0][2].close_to(embedding_layer.weights[2])
    assert result[0][3].close_to(embedding_layer.weights[0])

    assert result[1][0].close_to(embedding_layer.weights[1])
    assert result[1][1].close_to(embedding_layer.weights[2])
    assert result[1][2].close_to(embedding_layer.weights[2])
    assert result[1][3].close_to(embedding_layer.weights[1])

    result.backward()

    assert embedding_layer.weights.grad is not None
    assert embedding_layer.weights.grad.shape == (vocab_size, out_shape)
    assert embedding_layer.weights.grad.close_to(
        [
            [
                [2.0, 2.0, 2.0, 2.0, 2.0],
                [3.0, 3.0, 3.0, 3.0, 3.0],
                [3.0, 3.0, 3.0, 3.0, 3.0],
            ],
        ]
    )
