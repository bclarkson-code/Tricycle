import inspect

import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler

from llm_from_scratch.loss import categorical_crossentropy, mean_square_error
from llm_from_scratch.ops import einsum, no_grad, relu, tensor

np.random.seed(42)


class Dataset:
    """
    Iterator that generates batches from the dataset
    """

    def __init__(self, X, y, batch_size=32):
        self.X = X
        self.y = y
        self.batch_size = batch_size

    def __iter__(self):
        while True:
            indices = np.random.choice(len(self.X), self.batch_size)
            yield self.X[indices], self.y[indices]

    def __len__(self):
        return len(self.X)


def init_normal(shape, name: str = "", loc=0.0, scale=0.01):
    return tensor(np.random.normal(size=shape, loc=loc, scale=scale), name=name)


def init_zero(shape, name: str = ""):
    return tensor(np.zeros(shape), name=name)


def init_xavier(shape, name: str = ""):
    f_in, f_out = shape
    bound = np.sqrt(6) / np.sqrt(f_in + f_out)
    return tensor(np.random.uniform(low=-bound, high=bound, size=shape), name=name)


def test_simple_neural_network():
    """
    Train a simple neural network on the iris dataset
    """
    batch_size = 64
    learning_rate = 1e-0
    n_labels = 1
    n_features = 10
    layer_1_size = 16
    layer_2_size = 8

    diabetes = load_diabetes(scaled=True)
    x = diabetes.data

    # one_hot encode the labels
    y = diabetes.target.reshape(-1, 1)
    scaler = StandardScaler()
    y = scaler.fit_transform(y)

    # Build an iterator that generates random mini-batches.
    ds = Dataset(x, y, batch_size=batch_size)

    # define a model
    layer_1_weights = init_xavier((n_features, layer_1_size), name="layer_1_weights")
    layer_2_weights = init_xavier((layer_1_size, layer_2_size), name="layer_2_weights")
    layer_3_weights = init_xavier((layer_2_size, n_labels), name="layer_3_weights")

    layer_1_bias = init_zero((1, layer_1_size), name="layer_1_bias")
    layer_2_bias = init_zero((1, layer_2_size), name="layer_2_bias")
    layer_3_bias = init_zero((1, n_labels), name="layer_3_bias")

    params = [
        layer_1_weights,
        layer_2_weights,
        layer_3_weights,
        layer_1_bias,
        layer_2_bias,
        layer_3_bias,
    ]

    def model(x, params):
        (
            layer_1_weights,
            layer_2_weights,
            layer_3_weights,
            layer_1_bias,
            layer_2_bias,
            layer_3_bias,
        ) = params

        layer_1 = relu(
            einsum(x, layer_1_weights, subscripts="ij,jk->ik") + layer_1_bias
        )
        layer_2 = relu(
            einsum(layer_1, layer_2_weights, subscripts="ij,jk->ik") + layer_2_bias
        )
        layer_3 = (
            einsum(layer_2, layer_3_weights, subscripts="ij,jk->ik") + layer_3_bias
        )
        return layer_3

    losses = []

    for batch_idx, (X, y) in enumerate(ds):
        preds = model(X, params)
        loss = mean_square_error(preds, y)

        loss.backward()
        losses.append(loss)

        with no_grad():
            for idx, param in enumerate(params):
                params[idx] = param - (param.grad * learning_rate)

        if batch_idx >= 1000:
            break


    _, ax = plt.subplots(figsize=(16, 9))
    ax.plot(losses)
    plt.show()


if __name__ == "__main__":
    test_simple_neural_network()
