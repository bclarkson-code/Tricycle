import numpy as np

from llm_from_scratch.loss import categorical_crossentropy, mean_square_error
from llm_from_scratch.ops import einsum, no_grad, sigmoid, tensor


def test_mean_square_error():
    x = tensor(np.linspace(-1.0, 1.0, 100))

    # hard code slope = 2, intercept = -1
    y = tensor(np.linspace(-1.0, 1.0, 100) * 2 - 1)

    slope = tensor(0.01)  # start off with a small slope
    intercept = tensor(0.0)  # start off with a small intercept

    learning_rate = tensor(0.1)

    prev_loss = np.inf
    # sourcery skip: no-loop-in-tests
    for _ in range(100):
        expanded_intercept = einsum(
            intercept, np.ones_like(x), subscripts=",a->a"
        )

        z = einsum(x, slope, subscripts="a,->a") + expanded_intercept

        # Calculate mean squared error
        loss = mean_square_error(z, y)

        # Make sure loss is decreasing
        assert prev_loss > loss
        prev_loss = loss

        # Do backprop
        loss.backward()

        with no_grad():
            slope = slope - slope.grad * learning_rate
            intercept = intercept - intercept.grad * learning_rate

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
    x = [
        0.50,
        0.75,
        1.00,
        1.25,
        1.50,
        1.75,
        1.75,
        2.00,
        2.25,
        2.50,
        2.75,
        3.00,
        3.25,
        3.50,
        4.00,
        4.25,
        4.50,
        4.75,
        5.00,
        5.5,
    ]
    y = [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1]
    y = [[v, 1 - v] for v in y]

    x = tensor(x)
    y = tensor(y)

    slope = tensor([-0.05, 0.1])  # start off with a small slope
    intercept = tensor(0.01)  # start off with a small intercept

    learning_rate = tensor(1e-2)

    prev_loss = np.inf
    # sourcery skip: no-loop-in-tests
    for _ in range(100):
        z = sigmoid(einsum(slope, x, subscripts="i,j->ji") + intercept)

        # Calculate cross entropy
        loss = categorical_crossentropy(z, y)

        # Do backprop
        loss.backward()

        with no_grad():
            slope = slope - slope.grad * learning_rate
            intercept = intercept - intercept.grad * learning_rate

    # Make sure we are close to the true slope and intercept
    assert np.allclose(slope, 2.0, rtol=0.1)
    assert np.allclose(intercept, -1.0, rtol=0.1)
