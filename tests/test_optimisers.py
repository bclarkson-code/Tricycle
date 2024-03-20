import numpy as np
from sklearn.datasets import load_iris

from tricycle.activation import ReLU
from tricycle.dataset import InfiniteBatchDataset
from tricycle.layers import Dense, Sequential
from tricycle.loss import cross_entropy
from tricycle.optimisers import StochasticGradientDescent


def test_can_train_simple_neural_network_no_wd():
    """
    Train a simple neural network on the iris dataset
    """
    BATCH_SIZE = 16
    N_STEPS = 10

    np.random.seed(42)
    X, y = load_iris(return_X_y=True)
    # one hot encode y
    y = np.eye(3)[y.astype(int)]

    # create a dataset
    ds = InfiniteBatchDataset(X, y, batch_size=BATCH_SIZE)

    # create a model
    layer_1 = Dense(4, 16)
    layer_2 = Dense(16, 3)
    relu = ReLU()
    model = Sequential(layer_1, relu, layer_2)
    loss_fn = cross_entropy
    optimiser = StochasticGradientDescent(learning_rate=1e-2)

    losses = []
    batches = ds.to_tensor().to_vector()
    # sourcery skip: no-loop-in-tests
    # sourcery skip: no-conditionals-in-tests
    for step, (x, y) in enumerate(batches):
        if step > N_STEPS:
            break

        y_pred = model(x)
        loss = loss_fn(y, y_pred).from_vector().e("a->") / BATCH_SIZE
        loss.backward()
        losses.append(loss)

        model.update(optimiser)
        model.zero_grad()

    assert losses[-1] < 1.5


def test_can_train_simple_neural_network_wd():
    """
    Train a simple neural network on the iris dataset with weight decay
    """
    BATCH_SIZE = 16
    N_STEPS = 10

    np.random.seed(42)
    X, y = load_iris(return_X_y=True)
    # one hot encode y
    y = np.eye(3)[y.astype(int)]

    # create a dataset
    ds = InfiniteBatchDataset(X, y, batch_size=BATCH_SIZE)

    # create a model
    layer_1 = Dense(4, 16)
    layer_2 = Dense(16, 3)
    relu = ReLU()
    model = Sequential(layer_1, relu, layer_2)
    loss_fn = cross_entropy
    optimiser = StochasticGradientDescent(learning_rate=1e-2, weight_decay=1e1)

    losses = []
    batches = ds.to_tensor().to_vector()
    # sourcery skip: no-loop-in-tests
    # sourcery skip: no-conditionals-in-tests
    for step, (x, y) in enumerate(batches):
        if step > N_STEPS:
            break

        y_pred = model(x)
        loss = loss_fn(y, y_pred).from_vector().e("a->") / BATCH_SIZE
        loss.backward()
        losses.append(loss)

        model.update(optimiser)
        model.zero_grad()

    assert losses[-1] < 1.5


def test_can_train_simple_neural_network_momentum():
    """
    Train a simple neural network on the iris dataset with momentum
    """
    BATCH_SIZE = 16
    N_STEPS = 10

    np.random.seed(42)
    X, y = load_iris(return_X_y=True)
    # one hot encode y
    y = np.eye(3)[y.astype(int)]

    # create a dataset
    ds = InfiniteBatchDataset(X, y, batch_size=BATCH_SIZE)

    # create a model
    layer_1 = Dense(4, 16)
    layer_2 = Dense(16, 3)
    relu = ReLU()
    model = Sequential(layer_1, relu, layer_2)
    loss_fn = cross_entropy
    optimiser = StochasticGradientDescent(learning_rate=1e-2, momentum=0.9)

    losses = []
    batches = ds.to_tensor().to_vector()
    # sourcery skip: no-loop-in-tests
    # sourcery skip: no-conditionals-in-tests
    for step, (x, y) in enumerate(batches):
        if step > N_STEPS:
            break

        y_pred = model(x)
        loss = loss_fn(y, y_pred).from_vector().e("a->") / BATCH_SIZE
        loss.backward()
        losses.append(loss)

        model.update(optimiser)
        model.zero_grad()

    assert losses[-1] < 1.5
