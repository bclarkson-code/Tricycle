import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler

from tricycle.activation import relu
from tricycle.dataset import InfiniteDataset
from tricycle.experiments import smooth, plot_loss
from tricycle.initialisers import init_xavier, init_zero
from tricycle.loss import mean_square_error
from tricycle.ops import einsum, no_grad


def test_simple_neural_network():
    """
    Train a simple regression network on the diabetes dataset
    """
    # Make sure our results are reproducible
    np.random.seed(42)

    batch_size = 64
    learning_rate = 1e-0
    n_epochs = 1000
    n_labels = 1
    n_features = 10
    layer_1_size = 16
    layer_2_size = 8

    diabetes = load_diabetes(scaled=True)
    x = diabetes.data

    # We need to scale the target to avoid giant grads
    y = diabetes.target.reshape(-1, 1)
    y = StandardScaler().fit_transform(y)

    # Build an iterator that generates random mini-batches.
    ds = InfiniteDataset(x, y, batch_size=batch_size)

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
                param.grad = None
                

        # sourcery skip: no-conditionals-in-tests
        if batch_idx >= n_epochs:
            break

    # Check that our loss has actually decreased
    # It starts at about 0.9 and drops to around 0.5
    # if the model actually learns
    plot_loss(losses)
    assert list(smooth(losses))[-1] < 0.6
