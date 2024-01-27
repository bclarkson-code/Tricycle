import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris

from tricycle.activation import ReLU
from tricycle.dataset import Dataset
from tricycle.layers import Dense, Sequential
from tricycle.loss import cross_entropy


def test_can_train_simple_neural_network():
    """
    Train a simple neural network on the iris dataset
    """
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

    BATCH_SIZE = 16
    LEARNING_RATE = 1e-2 / BATCH_SIZE
    N_EPOCHS = 10

    losses = []
    for _ in range(N_EPOCHS):
        batches = ds.copy().to_tensor().shuffle().batch(BATCH_SIZE)
        for batch in batches:
            if not batch or len(batch[0]) != BATCH_SIZE:
                continue
            total_loss = 0
            for x, y in zip(*batch):
                x.name = "x"
                y.name = "y"
                y_pred = model(x)
                loss = cross_entropy(y, y_pred)
                # loss.show_graph = True
                loss.backward()
                total_loss += loss
            losses.append(total_loss / BATCH_SIZE)
            model.update(LEARNING_RATE)
            model.zero_grad()

    # plt.plot(losses)
    # plt.show()
