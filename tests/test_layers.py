import numpy as np

from tricycle.layers import Dense, Sequential


def test_dense_layer():
    layer = Dense(10, 8)

    assert layer.weights.shape == (10, 8)

    x_in = np.ones(10)

    x_out = layer(x_in)
    assert x_out.shape == (8,)


def test_sequential_layer():
    layer1 = Dense(10, 8)
    layer2 = Dense(8, 4)

    seq = Sequential(layer1, layer2)

    assert seq.layers[0].weights.shape == (10, 8)
    assert seq.layers[1].weights.shape == (8, 4)

    x_in = np.ones(10)

    x_out = seq(x_in)
    assert x_out.shape == (4,)
