import random
from typing import Sequence

import numpy as np

from tricycle.tensor import to_tensor


class Dataset:
    """
    An in-memory dataset: not suitable for large datasets
    """

    def __init__(self, inputs: Sequence, outputs: Sequence):
        assert len(inputs) == len(outputs)
        self.inputs = inputs
        self.outputs = outputs
        self._indices = list(range(len(inputs)))
        self._index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._index >= len(self.inputs):
            raise StopIteration

        result = self[self._index]
        self._index += 1
        return result

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx: int):
        idx = self._indices[idx]
        return self.inputs[idx], self.outputs[idx]

    def shuffle(self):
        np.random.shuffle(self._indices)
        return self

    def to_tensor(self):
        self.inputs = [to_tensor(x) for x in self.inputs]
        self.outputs = [to_tensor(x) for x in self.outputs]
        return self

    def reset(self):
        self._index = 0
        return self

    def copy(self):
        return Dataset(self.inputs.copy(), self.outputs.copy())


class InfiniteBatchDataset(Dataset):
    is_infinite = True
    _to_tensor = False
    is_vector = False

    def __init__(self, inputs: Sequence, outputs: Sequence, batch_size: int):
        super().__init__(inputs, outputs)
        self.batch_size = batch_size

    def __next__(self):
        result = self[self._index]
        self._index += 1
        return result

    def __len__(self):
        return -1

    def __getitem__(self, idx: int):
        random.seed(idx)
        indices = [
            random.randint(0, len(self.inputs) - 1)
            for _ in range(self.batch_size)
        ]
        batch_inputs = np.vstack([self.inputs[i] for i in indices])
        batch_outputs = np.vstack([self.outputs[i] for i in indices])

        if self._to_tensor:
            batch_inputs = to_tensor(batch_inputs, is_vector=self.is_vector)
            batch_outputs = to_tensor(batch_outputs, is_vector=self.is_vector)
        return batch_inputs, batch_outputs

    def to_tensor(self):
        self._to_tensor = True
        return self

    def to_vector(self):
        self.is_vector = True
        return self

    def from_vector(self):
        self.is_vector = False
        return self
