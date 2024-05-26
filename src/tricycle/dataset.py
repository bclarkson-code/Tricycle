import random
from typing import Sequence

import numpy as np
from sklearn.datasets import fetch_olivetti_faces

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
            batch_inputs = to_tensor(
                batch_inputs,
                is_vector=self.is_vector,
                dtype=batch_outputs.dtype,
            )
            batch_outputs = to_tensor(
                batch_outputs,
                is_vector=self.is_vector,
                dtype=batch_outputs.dtype,
            )
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


class CausalLMDataset:
    tokens: Sequence[int]
    vocab_size: int
    context_window: int

    batch_size: int | None = None
    is_batch: bool = False
    batch_indices: np.ndarray | None = None
    device: int | None = None

    _idx: int = 0

    def __init__(
        self,
        tokens: Sequence[int],
        vocab_size: int,
        context_window: int,
    ):
        self.tokens = tokens
        self.vocab_size = vocab_size
        self.context_window = context_window
        self._idx = 0

    def __len__(self):
        if self.batched:
            assert self.batch_size
            return (
                len(self.tokens) - self.context_window - self.batch_size - 1
            ) // self.batch_size
        return len(self.tokens) - 1

    def _get_single(self, idx: int):
        """
        Get a single input-output pair
        """
        if idx >= len(self.tokens) - self.context_window - 1:
            raise IndexError(f"Index {idx} out of range")

        tokens = self.tokens[idx : idx + self.context_window + 1]
        inputs = tokens[:-1]
        outputs = tokens[1:]
        return inputs, outputs

    def _get_batch(self, idx: int):
        """
        Get a batch of input-output pairs
        """
        assert self.batch_size is not None
        assert self.batch_indices is not None

        if idx >= len(self.tokens) - self.context_window - self.batch_size - 1:
            raise IndexError(f"Index {idx} out of range")

        start = idx * self.batch_size
        end = start + self.batch_size
        batch_indices = self.batch_indices[start:end]

        tokens = np.array(
            [
                self.tokens[batch_idx : batch_idx + self.context_window + 1]
                for batch_idx in batch_indices
            ]
        )
        return tokens[:, :-1], tokens[:, 1:]

    def __getitem__(self, idx: int):
        inputs, output = (
            self._get_batch(idx) if self.batched else self._get_single(idx)
        )
        inputs = to_tensor(
            inputs, requires_grad=False, name="inputs", dtype=np.int32
        )
        output = to_tensor(
            output, requires_grad=False, name="output", dtype=np.int32
        )

        if self.batched:
            inputs = inputs.to_vector()
            output = output.to_vector()

        if self.use_gpu:
            assert self.device is not None
            inputs = inputs.to_gpu(self.device)
            output = output.to_gpu(self.device)

        return inputs, output

    def __iter__(self):
        self._idx = 0
        return self

    def __next__(self):
        if self._idx >= len(self):
            raise StopIteration

        result = self[self._idx]
        self._idx += 1
        return result

    def batch(self, batch_size: int):
        self.batch_size = batch_size
        self.batched = True
        self.batch_indices = np.arange(
            len(self.tokens) - self.context_window - 1
        )
        return self

    def unbatch(self):
        self.batched = False
        return self

    def shuffle(self):
        if not self.batched:
            raise NotImplementedError(
                "Shuffling non-batched datasets is not currently supported"
            )
        else:
            assert self.batch_indices is not None
            self.batch_indices = np.random.choice(
                self.batch_indices, replace=False, size=len(self.batch_indices)
            )
        return self

    def to_gpu(self, device: int = 0):
        self.device = device
        self.use_gpu = True
        return self

    def from_gpu(self):
        self.device = None
        self.use_gpu = False
        return self
