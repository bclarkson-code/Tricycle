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

    def batch(self, size: int):
        batched_inputs = []
        batched_outputs = []

        idx = 0
        while idx < len(self.inputs):
            input_batch = []
            output_batch = []
            for _ in range(size):
                if idx >= len(self.inputs):
                    break
                inp, out = self[idx]
                input_batch.append(inp)
                output_batch.append(out)
                idx += 1
            batched_inputs.append(input_batch)
            batched_outputs.append(output_batch)
        self.inputs = batched_inputs
        self.outputs = batched_outputs
        self._indices = list(range(len(batched_inputs)))
        return self

    def copy(self):
        return Dataset(self.inputs.copy(), self.outputs.copy())
