import numpy as np
import pytest
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris

from tricycle.activation import ReLU
from tricycle.dataset import Dataset
from tricycle.layers import Dense, Sequential
from tricycle.loss import cross_entropy
from tricycle.optimisers import StochasticGradientDescent
from tricycle.reduce import radd
from tricycle.tensor import to_tensor


@pytest.mark.skip
def test_can_train_simple_neural_network():
    """
    Train a simple neural network on the iris dataset
    """
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-2
    N_EPOCHS = 10

    np.random.seed(42)
    X, y = load_iris(return_X_y=True)
    # one hot encode y
    y = np.eye(3)[y.astype(int)]

    # create a dataset
    ds = Dataset(X, y)

    # create a model
    layer_1 = Dense(4, 16)
    layer_2 = Dense(16, 3)
    relu = ReLU()
    model = Sequential(layer_1, relu, layer_2).vectorise()
    loss_fn = cross_entropy().vectorise()
    optimiser = StochasticGradientDescent(learning_rate=LEARNING_RATE)

    losses = []
    for _ in range(N_EPOCHS):
        batches = ds.copy().to_tensor().shuffle().batch(BATCH_SIZE)
        for x, y in batches:
            x = to_tensor(x)
            y = to_tensor(y)

            y_pred = model(x)
            loss = loss_fn(y, y_pred)
            loss = radd(loss / len(x), "i->")
            loss.backward()
            losses.append(loss)

            model.update(optimiser)
            model.zero_grad()

    plt.plot(losses)
    plt.show()
