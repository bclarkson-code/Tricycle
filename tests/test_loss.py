import numpy as np
import pytest
from sklearn.datasets import load_diabetes, load_iris, load_linnerud
from sklearn.preprocessing import RobustScaler

from tricycle.einsum import Einsum
from tricycle.initialisers import init_xavier
from tricycle.loss import CrossEntropy, MeanSquaredError
from tricycle.tensor import Tensor
from tricycle.utils import r_squared, smooth

slow_test = pytest.mark.skipif(
    "not config.getoption('--run-slow')",
    reason="Only run when --run-slow is given",
)

# TODO: a lot of these tests are outdated and should be modernised


def test_can_mean_square_error():
    y_true = Tensor([0, 0, 1])
    y_pred = Tensor([0, 0.5, 0.5])

    mse = MeanSquaredError()(y_true, y_pred)

    assert mse.close_to(1 / 6, rtol=1e-3)


def test_can_CrossEntropy():
    y_true = Tensor([1], dtype=int)
    y_pred = Tensor([[0, 0, 0]])

    loss = CrossEntropy()(y_true, y_pred)

    assert loss.close_to(1.0986122886681098)


def test_CrossEntropy_batched():
    batch_size = 3
    n_tokens = 5
    vocab_size = 7

    y_true = np.random.randint(0, vocab_size, size=(batch_size, n_tokens))
    y_pred = np.random.random((batch_size, n_tokens, vocab_size))

    y_true = Tensor(y_true, dtype=int).to_batched()
    y_pred = Tensor(y_pred).to_batched()

    loss = CrossEntropy()(y_true, y_pred)

    assert loss.shape == ()


# TODO: write a proper backprop test for these loss functions


def test_can_single_linear_regression_step():
    """
    A single step of linear regression
    """
    x_input = [1]
    y_input = [3]
    slope = Tensor([0.02])
    intercept = Tensor([0.01])

    x_input = Tensor(x_input, requires_grad=False, name="x")
    y_input = Tensor(y_input, requires_grad=False, name="y")

    y_pred = x_input * slope + intercept

    loss = MeanSquaredError()(y_input, y_pred)

    assert loss.close_to(8.8209, rtol=1e-3)

    loss.backward()
    assert slope.grad is not None
    assert intercept.grad is not None
    assert slope.grad.close_to([-5.94], rtol=1e-3)
    assert intercept.grad.close_to([-5.94], rtol=1e-3)


@slow_test
def test_can_linear_regression():
    np.random.seed(42)

    n = 4
    learning_rate = 3e-3
    learning_rate /= n
    x = np.linspace(-10, 10, n)
    y = x * 2 + 1 + np.random.normal(loc=0, scale=0.01, size=n)

    x = Tensor(x.reshape(-1, 1), requires_grad=False, name="x")
    y = Tensor(y.reshape(-1, 1), requires_grad=False, name="y")

    slope = Tensor([0.02], name="slope")
    intercept = Tensor([0.0], name="intercept")

    def slope_derivative(x, y, slope, intercept):
        return -2 * (y - x * slope - intercept) * x

    def intercept_derivative(x, y, slope, intercept):
        return -2 * (y - x * slope - intercept)

    losses = [0] * 100
    intercepts = []
    slopes = []
    for idx in range(100):
        last_slope_grad = Tensor([0])
        last_intercept_grad = Tensor([0])
        for x_input, y_input in zip(x, y):
            x_input = Tensor(x_input, requires_grad=False, name="x")
            y_input = Tensor(y_input, requires_grad=False, name="y")
            y_pred = x_input * slope + intercept
            loss = MeanSquaredError()(y_input, y_pred)
            losses[idx] += loss

            loss.backward()

            slope_grad = slope_derivative(x_input, y_input, slope, intercept)
            intercept_grad = intercept_derivative(
                x_input, y_input, slope, intercept
            )

            assert slope.grad is not None
            assert intercept.grad is not None

            assert slope.grad.close_to(
                last_slope_grad + slope_grad, atol=1e-4
            ), f"{slope.grad=}, {last_slope_grad=}, {slope_grad=}"

            assert intercept.grad.close_to(
                last_intercept_grad + intercept_grad
            ), f"{intercept.grad=}, {last_intercept_grad=}, {intercept_grad=}"

            last_slope_grad = slope.grad
            last_intercept_grad = intercept.grad

        slope -= slope.grad * learning_rate
        intercept -= intercept.grad * learning_rate

        slopes.append(slope)
        intercepts.append(intercept)
        slope = slope.zero_grad()
        intercept = intercept.zero_grad()

    assert losses[-1] < 1.5
    assert slopes[-1].close_to(2, atol=0.01)
    # The intercept takes much longer to tune
    assert intercepts[-1].close_to(0.455, atol=0.01)


@slow_test
def test_linear_regression_real_data():
    X, y = load_diabetes(return_X_y=True)
    x_scaler = RobustScaler()
    y_scaler = RobustScaler()
    X = x_scaler.fit_transform(X)
    y = y_scaler.fit_transform(y.reshape(-1, 1))

    X = Tensor(X)
    y = Tensor(y)

    loops = 100
    learning_rate = 1e-1
    n = len(X)
    learning_rate /= n

    slope = init_xavier((X.shape[1], 1))
    intercept = Tensor([0], name="intercept")

    for _ in range(loops):
        for x_in, y_in in zip(X, y):
            y_pred = Einsum("i,ij->j")(x_in, slope) + intercept
            loss = mean_square_error(y_in, y_pred)
            loss.backward()

        slope = Tensor(slope - slope.grad * learning_rate, name="slope")
        intercept = Tensor(
            intercept - intercept.grad * learning_rate, name="intercept"
        )

    predicted = X @ np.array(slope) + intercept[0]
    r_square = r_squared(np.array(y), predicted)
    assert r_square > 0.45


@slow_test
def test_linear_regression_multi_input_output():
    X_data, y_data = load_linnerud(return_X_y=True)
    x_scaler = RobustScaler()
    y_scaler = RobustScaler()
    X_data = x_scaler.fit_transform(X_data)
    y_data = y_scaler.fit_transform(y_data)

    learning_rate = 1e-0
    n = len(X_data)
    learning_rate /= n
    loops = 100

    slope = init_xavier((X_data.shape[1], y_data.shape[1]), name="slope")
    intercept = Tensor([-0.01, 0.01, 0.02], name="intercept")

    losses: list[np.ndarray | int] = [0] * loops

    def model(X, slope, intercept):
        return Einsum("i,ij->j")(X, slope) + intercept

    for idx in range(loops):
        X = Tensor(X_data).to_batched()
        y = Tensor(y_data).to_batched()

        # predict an output
        y_pred = model(X, slope, intercept)

        # calculate the loss
        loss = mean_square_error(y, y_pred)
        # we need to unbatch the loss before finding its average
        loss = loss.from_batched().mean()

        losses[idx] = loss.numpy()
        loss.backward()

        assert slope.grad is not None
        assert intercept.grad is not None

        slope.grad = slope.grad.from_batched().einsum("abc->bc")
        intercept.grad = intercept.grad.from_batched().einsum("ab->b")

        slope = (slope - slope.grad * learning_rate).zero_grad()
        intercept = (intercept - intercept.grad * learning_rate).zero_grad()

    # the loss should plateau at around 0.5
    assert losses[-1] < 0.6


@slow_test
def test_CrossEntropy():
    """
    This is a really slow test, preserved for reference
    """
    X, y = load_iris(return_X_y=True)
    x_scaler = RobustScaler()
    X = x_scaler.fit_transform(X)

    # one hot encode y
    y = np.eye(3)[y.astype(int)]

    X = Tensor(X)
    y = Tensor(y)

    learning_rate = 1e0
    n = len(X)
    learning_rate /= n
    loops = 100

    slope = init_xavier((X.shape[1], y.shape[1]), name="slope")
    intercept = Tensor([-0.01, 0.01, 0.02], name="intercept")

    losses = [0] * loops
    for idx in range(loops):
        for x_in, y_in in zip(X, y):
            y_pred = Einsum("i,ij->j")(x_in, slope) + intercept
            loss = CrossEntropy()(y_in, y_pred)
            losses[idx] += loss
            loss.backward()

        slope = Tensor(slope - slope.grad * learning_rate, name="slope")
        intercept = Tensor(
            intercept - intercept.grad * learning_rate, name="intercept"
        )

    assert losses[-1] < 35


@slow_test
def test_CrossEntropy_minibatch():
    """
    This is a really slow test, preserved for reference
    """
    np.random.seed(42)

    def dataset(X, y, batch_size):
        while True:
            indices = np.arange(len(X))
            batch_indices = np.random.choice(
                indices, size=batch_size, replace=False
            )
            yield zip(X[batch_indices], y[batch_indices])

    X, y = load_iris(return_X_y=True)
    x_scaler = RobustScaler()
    X = x_scaler.fit_transform(X)
    y = np.eye(3)[y.astype(int)]

    learning_rate = 1e0
    learning_rate /= 16
    loops = 500

    slope = init_xavier((X.shape[1], y.shape[1]), name="slope")
    intercept = Tensor([-0.01, 0.01, 0.02], name="intercept")

    losses = []
    for idx, batch in enumerate(dataset(X, y, batch_size=16)):
        if idx > loops:
            break
        batch_loss = 0
        for x_in, y_in in batch:
            x_in = Tensor(x_in)
            y_in = Tensor(y_in)

            y_pred = Einsum("i,ij->j")(x_in, slope) + intercept
            loss = CrossEntropy()(y_in, y_pred)
            batch_loss += loss
            loss.backward()

        losses.append(batch_loss)

        slope = Tensor(slope - slope.grad * learning_rate, name="slope")
        intercept = Tensor(
            intercept - intercept.grad * learning_rate, name="intercept"
        )

    losses = list(smooth(losses, 0.99))
    assert losses[-1] < 6
