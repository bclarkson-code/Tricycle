import numpy as np

from tricycle_v2.loss import mean_squared_error
from tricycle_v2.tensor import to_tensor
from tricycle_v2.ops import repeat 


def test_can_mean_square_error():
    y_true = to_tensor([[0, 0, 1], [0, 1, 0], [1 / 3, 1 / 3, 1 / 3]])
    y_pred = to_tensor([[0, 0, 1], [0, 0, 1], [0, 0, 1]])

    mse = mean_squared_error(y_true, y_pred)

    assert mse.shape == (3,)
    assert np.allclose(mse, np.array([0, 2 / 3, 2 / 9]))


def test_can_linear_regression():
    np.random.seed(42)

    x = np.linspace(-10, 10, 201)
    y = x * 2 + 1 + np.random.normal(loc=0, scale=0.01, size=201)

    x = to_tensor(x.reshape(-1, 1))
    y = to_tensor(y)

    slope = to_tensor([0.01])
    intercept = to_tensor(0.01)

    for _ in range(100):
        repeated_slope = repeat("i->ji", slope, (x.shape[0],))
        repeated_intercept = repeat("i->ji", intercept, (x.shape[0],))

        y_pred = x * repeated_slope + repeated_intercept
        loss = mean_squared_error(y, y_pred)

        loss.backward()
        breakpoint()

        slope -= slope.grad
        intercept -= intercept.grad
