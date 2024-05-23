import time
from tricycle import CUPY_ENABLED
from tricycle.exceptions import GPUDisabledException
from abc import abstractmethod
from pathlib import Path
from typing import Iterable

import humanize
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


def log_memory_and_time(stage: str, path: Path = Path("memory.log")):
    """
    Log the current GPU memory usage to a file
    """
    if not CUPY_ENABLED:
        raise GPUDisabledException("Cannot log GPU memory if GPU is not enabled")

    import cupy

    if not path.exists():
        path.write_text(
            "stage,used_bytes_human,total_bytes_human,used_bytes,total_bytes,timestamp\n"  # noqa: E501
        )

    pool = cupy.get_default_memory_pool()
    now = time.perf_counter()

    used_bytes = humanize.naturalsize(pool.used_bytes())
    total_bytes = humanize.naturalsize(pool.total_bytes())
    with open(path, "a") as f:
        f.write(
            f"{stage},{used_bytes},{total_bytes},{pool.used_bytes()},{pool.total_bytes()},{now}\n"  # noqa: E501
        )
