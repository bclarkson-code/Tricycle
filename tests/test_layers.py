from copy import copy

import numpy as np

from tricycle.layers import Dense, Dropout, LayerNorm, Sequential
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
