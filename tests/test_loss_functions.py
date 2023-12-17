import numpy as np

from llm_from_scratch.loss import categorical_crossentropy, mean_square_error
from llm_from_scratch.ops import sigmoid, sub, tensor


def test_mean_square_error():
    x = tensor(np.linspace(-1.0, 1.0, 100))

    # hard code slope = 2, intercept = -1
    y = tensor(np.linspace(-1.0, 1.0, 100) * 2 - 1)

    slope = tensor(0.01)  # start off with a small slope
    intercept = tensor(0.0)  # start off with a small intercept

    learning_rate = tensor(0.1)

    prev_loss = np.inf
    for _ in range(100):
        z = slope * x + intercept

        # Calculate mean squared error
        loss = mean_square_error(z, y)

        # Make sure loss is decreasing
        assert prev_loss > loss
        prev_loss = loss
        print(loss)

        # Do backprop
        loss.backward()

        slope = sub(slope, slope.grad * learning_rate, grad=False)
        intercept = sub(intercept, intercept.grad * learning_rate, grad=False)

    # Make sure we are close to the true slope and intercept
    assert np.allclose(slope, 2.0, rtol=0.1)
    assert np.allclose(intercept, -1.0, rtol=0.1)


def test_cross_entropy():
    """
    We'll test cross entropy loss by fitting a logistic
    regression to samples from the function
    if x > 0:
        return 1
    else:
        return 0
    """

    x = np.expand_dims(np.linspace(-1.0, 1.0, 5), 0).T
    y = np.array([(x < 0).astype(int), (x > 0).astype(int)])
    y = np.squeeze(y).T

    x = tensor(x)
    y = tensor(y)

    slope = tensor(
        np.expand_dims(np.array([0.01, 0.01]), 1)
    )  # start off with a small slope
    intercept = tensor(0.0)  # start off with a small intercept

    learning_rate = tensor(0.01)

    prev_loss = np.inf
    for _ in range(100):
        z = sigmoid(slope @ x.T).T

        # Calculate cross entropy
        loss = categorical_crossentropy(z, y)

        # Make sure loss is decreasing
        # assert prev_loss > loss
        prev_loss = loss

        print(loss)
        print(slope)
        print(slope.grad)

        # Do backprop
        loss.backward()

        slope = sub(slope, slope.grad * learning_rate, grad=False)
        intercept = sub(intercept, intercept.grad * learning_rate, grad=False)

    # Make sure we are close to the true slope and intercept
    assert np.allclose(slope, 2.0, rtol=0.1)
    assert np.allclose(intercept, -1.0, rtol=0.1)
