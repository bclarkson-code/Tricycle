import numpy as np
from sklearn.datasets import load_iris

from tricycle.experiments import plot_loss
from tricycle.initialisers import init_xavier
from tricycle.loss import categorical_crossentropy, mean_square_error
from tricycle.ops import einsum, no_grad, sigmoid, tensor


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
        expanded_intercept = einsum(intercept, np.ones_like(x), subscripts=",a->a")

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


# This does not currently work and I haven't figured out why yet
def test_cross_entropy():
    """
    We'll test cross entropy loss by fitting a single layer neural
    network to the iris dataset
    """
    iris = load_iris()
    x = tensor(iris.data, name="x", requires_grad=False)

    # one hot encode the labels
    y = tensor(
        np.eye(iris.target.max() + 1, dtype=np.float32)[iris.target],
        name="y",
        requires_grad=False,
    )

    slope = init_xavier(shape=(x.shape[1], y.shape[1]))  # start off with a small slope
    intercept = tensor(
        [-0.05, 0.01, 0.05], name="intercept"
    )  # start off with a small intercept

    learning_rate = tensor(1e0)
    learning_rate.requires_grad = False

    prev_loss = np.inf
    losses = []
    # sourcery skip: no-loop-in-tests
    for _ in range(10):
        print(f"{slope.shape=}, {x.shape=}, {y.shape=}")
        z = einsum(x, slope, subscripts="ij,jk->ik")
        ones = tensor(np.ones_like(z), requires_grad=False)
        broadcasted_intercept = einsum(
            intercept, ones, subscripts="k,jk->jk"
        )
        z += broadcasted_intercept
        z.name = "z"

        print(f"{z.shape=}")

        # Calculate cross entropy
        loss = categorical_crossentropy(z, y)
        losses.append(loss)
        assert prev_loss > loss

        # Do backprop
        loss.backward()

        with no_grad():
            raise Exception(slope.grad)
            slope = slope - slope.grad * learning_rate
            slope.grad = None
            intercept = intercept - intercept.grad * learning_rate
            intercept.grad = None

    plot_loss(losses)
