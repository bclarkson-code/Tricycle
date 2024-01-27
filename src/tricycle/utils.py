from abc import abstractmethod
from typing import Iterable

import numpy as np


class Dataset:
    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, index):
        raise NotImplementedError

    def __iter__(self):
        for idx in range(self.__len__()):
            yield self.__getitem__(idx)

    def __next__(self):
        return self.__getitem__(next(iter(self)))


def smooth(iterable: Iterable, factor: float):
    """
    Use exponential smoothing to smooth an array
    """
    prev = 0
    for val in iterable:
        yield prev * factor + (val - prev) * factor
        prev = val



def r_squared(actual_values, predicted_values):
    """
    calculate R-squared metric.
    """
    actual_values = np.array(actual_values)
    predicted_values = np.array(predicted_values)

    mean_actual = np.mean(actual_values)
    tss = np.sum((actual_values - mean_actual) ** 2)
    rss = np.sum((actual_values - predicted_values) ** 2)

    return 1 - (rss / tss)

