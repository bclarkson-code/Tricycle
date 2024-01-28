import numpy as np

from tricycle.activation import ReLU
from tricycle.layers import Dense, Sequential
from tricycle.loss import CrossEntropy, MeanSquareError
from tricycle.ops import einsum
from tricycle.tensor import to_tensor


def test_can_vectorise_single_einsum():
    input_1 = np.arange(1, 4)
    input_2 = np.arange(2, 5)
    input_3 = np.arange(3, 6)

    op = einsum("i->")

    output_1 = op(input_1)
    output_2 = op(input_2)
    output_3 = op(input_3)

    assert output_1 == 6
    assert output_2 == 9
    assert output_3 == 12

    input_vector = np.array([input_1, input_2, input_3])
    op = einsum("zi->z")
    output_vector = op(input_vector)
    assert np.allclose(output_vector, np.array([6, 9, 12]))


def test_can_vectorise_entire_model():
    np.random.seed(42)
    layer_1 = Dense(4, 16)
    layer_2 = Dense(16, 3)
    relu = ReLU()
    model = Sequential(layer_1, relu, layer_2)

    input_1 = to_tensor(np.arange(1, 5))
    input_2 = to_tensor(np.arange(2, 6))
    input_3 = to_tensor(np.arange(3, 7))

    output_1 = model(input_1)
    output_2 = model(input_2)
    output_3 = model(input_3)

    model.vectorise()
    input_vector = to_tensor(np.array([input_1, input_2, input_3]))

    output_vector = model(input_vector)

    assert np.allclose(output_vector, np.array([output_1, output_2, output_3]))


def test_can_vectorise_mse():
    mse = MeanSquareError()

    y_true = to_tensor([0, 0, 1, 0])

    input_1 = to_tensor(np.arange(1, 5))
    input_2 = to_tensor(np.arange(2, 6))
    input_3 = to_tensor(np.arange(3, 7))

    output_1 = mse(y_true, input_1)
    output_2 = mse(y_true, input_2)
    output_3 = mse(y_true, input_3)

    mse = mse.vectorise()

    input_y_true = to_tensor(np.array([y_true, y_true, y_true]))
    input_vector = to_tensor(np.array([input_1, input_2, input_3]))

    output_vector = mse(input_y_true, input_vector)

    assert np.allclose(output_vector, np.array([output_1, output_2, output_3]))


def test_can_vectorise_cross_entropy():
    cross_entropy = CrossEntropy()

    y_true = to_tensor([0, 0, 1, 0])
    input_1 = to_tensor(np.arange(1, 5))
    input_2 = to_tensor(np.arange(2, 6))
    input_3 = to_tensor(np.arange(3, 7))

    output_1 = cross_entropy(y_true, input_1)
    output_2 = cross_entropy(y_true, input_2)
    output_3 = cross_entropy(y_true, input_3)

    cross_entropy = cross_entropy.vectorise()

    input_y_true = to_tensor(np.array([y_true, y_true, y_true]))
    input_vector = to_tensor(np.array([input_1, input_2, input_3]))

    output_vector = cross_entropy(input_y_true, input_vector)

    assert np.allclose(output_vector, np.array([output_1, output_2, output_3]))
