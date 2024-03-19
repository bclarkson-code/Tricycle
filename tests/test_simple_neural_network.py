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

slow_test = pytest.mark.skipif(
    "not config.getoption('--run-slow')",
    reason="Only run when --run-slow is given",
)


@slow_test
def test_can_train_simple_neural_network():
    """
    Train a simple neural network on the iris dataset
    """
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-4
    N_EPOCHS = 100

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
    model = Sequential(layer_1, relu, layer_2)
    loss_fn = cross_entropy
    optimiser = StochasticGradientDescent(learning_rate=LEARNING_RATE)

    losses = []
    batches = ds.copy().to_tensor().shuffle().batch(BATCH_SIZE)
    for _ in range(N_EPOCHS):
        for x_in, y_out in batches:
            x_in = to_tensor(x_in, requires_grad=False).to_vector()
            y_out = to_tensor(y_out, requires_grad=False).to_vector()

            y_pred = model(x_in)
            loss = loss_fn(y_out, y_pred).from_vector().e("a->")
            loss.backward()
            losses.append(loss)

            model.update(optimiser)
            model.zero_grad()
        batches = batches.reset().shuffle()

    plt.plot(losses)
    plt.show()
