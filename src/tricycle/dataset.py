import random
from typing import Sequence

import numpy as np
from sklearn.datasets import fetch_olivetti_faces

from tricycle.tensor import Tensor


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
        self.inputs = [Tensor(x) for x in self.inputs]
        self.outputs = [Tensor(x) for x in self.outputs]
        return self

    def reset(self):
        self._index = 0
        return self

    def copy(self):
        return Dataset(self.inputs.copy(), self.outputs.copy())


class InfiniteBatchDataset(Dataset):
    is_infinite = True
    _to_tensor = False
    is_batched = True

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
            batch_inputs = Tensor(
                batch_inputs,
                is_batched=self.is_batched,
                dtype=batch_outputs.dtype,
            )
            batch_outputs = Tensor(
                batch_outputs,
                is_batched=self.is_batched,
                dtype=batch_outputs.dtype,
            )
        return batch_inputs, batch_outputs

    def to_tensor(self):
        self._to_tensor = True
        return self


class CausalLMDataset:
    def __init__(
        self,
        tokens: np.ndarray,
        vocab_size: int,
        batch_size: int,
        context_window: int,
        should_one_hot_encode: bool = False,
    ):
        self.tokens = tokens
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.context_window = context_window
        self.is_batch = False
        self.as_tensor = False
        self._idx = 0
        self.batch_indices = None
        self.should_one_hot_encode = should_one_hot_encode
        self.device = None

    def __len__(self):
        return (
            (len(self.tokens) - self.context_window - self.batch_size - 1)
            // self.batch_size
            if self.is_batch
            else len(self.tokens) - 1
        )

    def __getitem__(self, idx: int):
        if self.is_batch:
            start = idx * self.batch_size
            end = (idx + 1) * self.batch_size
            indices = self.batch_indices[start:end]
            batches = [
                self.tokens[i : i + self.context_window + 1] for i in indices
            ]
            inputs = np.vstack([b[:-1] for b in batches])
            outputs = np.vstack([b[1:] for b in batches])
        else:
            start = idx * self.context_window
            end = (idx + 1) * self.context_window + 1
            tokens = self.tokens[start:end]
            inputs = tokens[:-1]
            outputs = tokens[1:]

        if self.as_tensor:
            inputs = Tensor(
                inputs,
                requires_grad=False,
                name="inputs",
                is_batched=self.is_batch,
                dtype=outputs.dtype,
            )
            outputs = Tensor(
                outputs,
                requires_grad=False,
                name="output",
                is_batched=self.is_batch,
                dtype=outputs.dtype,
            )
            if self.device is not None:
                inputs.to_gpu(self.device)
                outputs.to_gpu(self.device)

        return inputs, outputs

    def __iter__(self):
        self._idx = 0
        return self

    def __next__(self):
        if self._idx >= len(self):
            raise StopIteration

        result = self[self._idx]
        self._idx += 1
        return result

    def batch(self):
        print("batching")
        self.is_batch = True
        self.batch_indices = np.arange(
            len(self.tokens) - self.context_window - 1
        )
        return self

    def unbatch(self):
        self.is_batch = False
        return self

    def shuffle(self):
        print("shuffling")
        if not self.is_batch and self.batch_indices is not None:
            raise NotImplementedError(
                "Shuffling non-batched datasets is not currently supported"
            )
        else:
            n_batches = len(self.tokens) - self.context_window - 1
            self.batch_indices = np.random.choice(
                n_batches, size=n_batches, replace=False
            )
        return self

    def to_gpu(self, device: int = 0):
        self.device = device
        return self

    def from_gpu(self):
        self.device = None
        return self

    def to_tensor(self):
        print("converting to tensor")
        self.as_tensor = True
        return self
