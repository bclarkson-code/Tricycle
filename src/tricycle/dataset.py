import numpy as np


class InfiniteDataset:
    """
    Iterator that infinitely generates random batches from the dataset
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
