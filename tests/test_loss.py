import numpy as np
import pytest
from matplotlib import pyplot as plt
from sklearn.datasets import load_diabetes, load_linnerud
from sklearn.preprocessing import RobustScaler

from tricycle.initialisers import init_xavier
from tricycle.loss import cross_entropy, mean_squared_error
from tricycle.ops import einsum, repeat
from tricycle.reduce import radd
from tricycle.tensor import to_tensor


def test_can_mean_square_error():
    y_true = to_tensor([0, 0, 1])
    y_pred = to_tensor([0, 0.5, 0.5])

    mse = mean_squared_error(y_true, y_pred)

    assert mse == 0.5


def test_can_cross_entropy():
    y_true = to_tensor([0, 0, 1])
    y_pred = to_tensor([0, 0, 0])

    loss = cross_entropy(y_true, y_pred)
    assert loss == 1.0986122886681098


def test_can_linear_regression():
    np.random.seed(42)

    n = 10
    learning_rate = 1e-2
    x = np.linspace(-10, 10, n)
    y = x * 2 + 1 + np.random.normal(loc=0, scale=0.01, size=n)

    x = to_tensor(x.reshape(-1, 1), requires_grad=False, name="x")
    y = to_tensor(y.reshape(-1, 1), requires_grad=False, name="y")

    slope = to_tensor([0.01], name="slope")
    intercept = to_tensor([0.01], name="intercept")

    losses = [0] * 100
    for idx in range(100):
        for x_input, y_input in zip(x, y):
            y_pred = x_input * slope + intercept
            loss = mean_squared_error(y_input, y_pred) / len(y)
            losses[idx] += loss
            loss.backward()

            breakpoint()

        slope = to_tensor(slope - slope.grad * learning_rate, name="slope")
        intercept = to_tensor(
            intercept - intercept.grad * learning_rate, name="intercept"
        )

    _, ax = plt.subplots()
    ax.plot(losses)
    ax.set_yscale("log")
    plt.show()


@pytest.mark.skip
def test_linear_regression_multi_input():
    X, y = load_diabetes(return_X_y=True)
    x_scaler = RobustScaler()
    y_scaler = RobustScaler()
    X = x_scaler.fit_transform(X)
    y = y_scaler.fit_transform(y.reshape(-1, 1))

    X = to_tensor(X)
    y = to_tensor(y)

    learning_rate = 1e-1

    slope = init_xavier((X.shape[1], 1), name="slope")
    intercept = to_tensor([0], name="intercept")

    losses = []
    for _ in range(100):
        repeated_intercept = repeat("j->ij", intercept, (X.shape[0], 1))

        y_pred = einsum("ij,jk->ik", X, slope) + repeated_intercept
        mse = mean_squared_error(y, y_pred)
        loss = radd(mse, "i->") / y.shape[0]

        losses.append(loss)

        loss.backward()

        slope = to_tensor(slope - slope.grad * learning_rate, name="slope")
        intercept = to_tensor(
            intercept - intercept.grad * learning_rate, name="intercept"
        )

    _, ax = plt.subplots()
    ax.plot(losses)
    ax.set_yscale("log")
    plt.show()


@pytest.mark.skip
def test_linear_regression_multi_input_output():
    X, y = load_linnerud(return_X_y=True)
    x_scaler = RobustScaler()
    y_scaler = RobustScaler()
    X = x_scaler.fit_transform(X)
    y = y_scaler.fit_transform(y)

    X = to_tensor(X)
    y = to_tensor(y)

    learning_rate = 1e-1

    slope = init_xavier((X.shape[1], y.shape[1]), name="slope")
    intercept = to_tensor([-0.01, 0.01, 0.02], name="intercept")

    losses = []
    for _ in range(100):
        repeated_intercept = repeat("k->ik", intercept, (X.shape[0], y.shape[1]))

        y_pred = einsum("ij,jk->ik", X, slope) + repeated_intercept
        mse = mean_squared_error(y, y_pred)
        loss = radd(mse, "i->") / y.shape[0]

        losses.append(loss)

        loss.backward()

        slope = to_tensor(slope - slope.grad * learning_rate, name="slope")
        intercept = to_tensor(
            intercept - intercept.grad * learning_rate, name="intercept"
        )

    _, ax = plt.subplots()
    ax.plot(losses)
    ax.set_yscale("log")
    plt.show()
