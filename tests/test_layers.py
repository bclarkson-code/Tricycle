from copy import copy

import numpy as np

from tricycle.einsum import Einsum
from tricycle.layers import Dense, Dropout, Embedding, LayerNorm, Sequential
from tricycle.tensor import to_tensor


def test_dense_layer():
    layer = Dense(10, 8)

    assert layer.weights.shape == (10, 8)

    x_in = to_tensor(np.ones(10))

    x_out = layer(x_in)
    assert x_out.shape == (8,)


def test_sequential_layer():
    layer1 = Dense(10, 8)
    layer2 = Dense(8, 4)

    model = Sequential(layer1, layer2)

    assert model.layers[0].weights.shape == (10, 8)
    assert model.layers[1].weights.shape == (8, 4)

    x_in = to_tensor(np.ones(10))

    x_out = model(x_in)
    assert x_out.shape == (4,)


def test_dropout():  # sourcery skip: square-identity
    np.random.seed(0)
    size = 100
    dropout_prob = 0.3

    # non-vectorised
    in_tensor = to_tensor(
        np.random.normal(size=(size, size)), name="in_tensor"
    )
    dropout = Dropout(dropout_prob)

    out_tensor = dropout(in_tensor.to_vector())

    assert out_tensor.shape == in_tensor.shape
    zero_x_idx, zero_y_idx = np.where(out_tensor._data == 0)
    n_zeros = len(zero_x_idx)
    expected_n_zeros = int(size * size * dropout_prob)

    assert np.allclose(n_zeros, expected_n_zeros, rtol=1e-2)

    out_tensor.backward()

    assert in_tensor.grad is not None
    assert in_tensor.grad.shape == in_tensor.shape

    correct_grad = np.ones(in_tensor.shape)
    correct_grad[zero_x_idx, zero_y_idx] = 0

    assert in_tensor.grad.close_to(correct_grad)


def test_layer_norm():
    np.random.seed(0)
    in_tensor = to_tensor(np.random.normal(size=(100, 100)), name="in_tensor")
    layer_norm = LayerNorm()
    out_tensor = layer_norm(in_tensor.to_vector())

    assert out_tensor.shape == in_tensor.shape
    out_tensor.backward()

    assert copy(out_tensor).mean().close_to([0] * 100, atol=1e-7)
    assert copy(out_tensor).standard_deviation().close_to([1] * 100, atol=1e-7)

    assert in_tensor.grad is not None
    assert in_tensor.grad.shape == in_tensor.shape

    # not sure if this is correct. TODO: check
    assert in_tensor.grad.close_to(np.zeros(in_tensor.shape), atol=1e-6)


def test_embedding():
    np.random.seed(0)
    vocab_size = 3
    out_shape = 5
    in_tensor = to_tensor(
        [0, 1, 2, 0],
        requires_grad=False,
        dtype=int,
    )

    embedding_layer = Embedding(from_size=vocab_size, to_size=out_shape)
    weights = np.indices((vocab_size * out_shape,)).reshape(
        vocab_size, out_shape
    )
    embedding_layer.weights = to_tensor(weights)

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


def test_embedding_vectorised():
    np.random.seed(0)
    vocab_size = 3
    out_shape = 5
    in_tensor = to_tensor(
        [[0, 1, 2, 0], [1, 2, 2, 1]],
        requires_grad=False,
        dtype=np.int8,
    ).to_vector()

    embedding_layer = Embedding(from_size=vocab_size, to_size=out_shape)
    weights = np.indices((vocab_size * out_shape,)).reshape(
        vocab_size, out_shape
    )
    embedding_layer.weights = to_tensor(weights)

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
    assert embedding_layer.weights.grad.shape == (2, vocab_size, out_shape)
    assert embedding_layer.weights.grad.close_to(
        [
            [
                [2.0, 2.0, 2.0, 2.0, 2.0],
                [1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0],
            ],
            [
                [0, 0, 0, 0, 0],
                [2.0, 2.0, 2.0, 2.0, 2.0],
                [2.0, 2.0, 2.0, 2.0, 2.0],
            ],
        ]
    )


def test_embedding_matches_orignal_method():
    vocab_size = 1024
    embed_dim = 384
    weights = to_tensor(np.random.random((vocab_size, embed_dim)))

    def original_embed(tokens):
        one_hot = np.zeros((tokens.shape[0], vocab_size))
        for i, token in enumerate(tokens._data):
            one_hot[i, token] = 1
        one_hot = to_tensor(one_hot)
        return Einsum("ca,aB->cB")(one_hot, weights)

    embedding_layer = Embedding(from_size=vocab_size, to_size=embed_dim)
    embedding_layer.weights = copy(weights)

    tokens = np.random.randint(low=0, high=1024, size=100)
    tokens = to_tensor(tokens, dtype=int, requires_grad=False)

    # 3.84 ms ± 17.6 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
    original_out = original_embed(tokens)
    # 28.1 µs ± 64.5 ns per loop (mean ± std. dev. of 7 runs, 10,000 loops each)
    new_out = embedding_layer(tokens)

    assert original_out.close_to(new_out)
