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


class CausalLMDataset:
    def __init__(
        self,
        tokens: Sequence[int],
        vocab_size: int,
        batch_size: int,
        context_window: int,
    ):
        self.tokens = tokens
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.context_window = context_window
        self.is_batch = False
        self._idx = 0
        self.batch_indices = None

    def __len__(self):
        return (
            (len(self.tokens) - self.context_window - self.batch_size - 1)
            // self.batch_size
            if self.is_batch
            else len(self.tokens) - 1
        )

    def one_hot_encode(self, tokens: Sequence[int]):
        """
        One hot encode some tokens into one-hot vectors
        """
        one_hot = np.zeros((len(tokens), self.vocab_size))

        for i, token in enumerate(tokens):
            one_hot[i, token] = 1
        return one_hot

    def _get_single(self, idx: int):
        """
        Get a single input-output pair
        """
        if idx >= len(self.tokens) - self.context_window - 1:
            raise IndexError(f"Index {idx} out of range")

        tokens = self.tokens[idx : idx + self.context_window + 1]
        encoded_tokens = self.one_hot_encode(tokens)
        inputs = tokens[:-1]
        outputs = encoded_tokens[1:]
        return inputs, outputs

    def _get_batch(self, idx: int):
        """
        Get a batch of input-output pairs
        """
        if idx >= len(self.tokens) - self.context_window - self.batch_size - 1:
            raise IndexError(f"Index {idx} out of range")

        start = idx * self.batch_size
        end = start + self.batch_size
        batch_indices = self.batch_indices[start:end]

        tokens = [
            self.tokens[batch_idx : batch_idx + self.context_window + 1]
            for batch_idx in batch_indices
        ]
        encoded_tokens = [self.one_hot_encode(t) for t in tokens]

        inputs = []
        outputs = []
        for token, encoded_token in zip(tokens, encoded_tokens):
            inputs.append(token[:-1])
            outputs.append(encoded_token[1:])

        inputs = np.array(inputs)
        outputs = np.array(outputs)
        return inputs, outputs

    def __getitem__(self, idx: int):
        inputs, output = (
            self._get_batch(idx) if self.is_batch else self._get_single(idx)
        )
        if self.as_tensor:
            inputs = to_tensor(
                inputs, requires_grad=False, name="inputs", dtype=int
            )
            output = to_tensor(output, requires_grad=False, name="output")

        if self.is_vector and not self.as_tensor:
            raise ValueError("Cannot vectorise an unbatched dataset")

        inputs = inputs.to_vector()
        output = output.to_vector()

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

    def batch(self):
        self.is_batch = True
        self.batch_indices = list(
            range(len(self.tokens) - self.context_window - 1)
        )
        return self

    def unbatch(self):
        self.is_batch = False
        return self

    def shuffle(self):
        if not self.is_batch and self.batch_indices is not None:
            raise NotImplementedError(
                "Shuffling non-batched datasets is not currently supported"
            )
        else:
            self.batch_indices = np.random.choice(
                self.batch_indices, replace=False, size=len(self.batch_indices)
            )
        return self

    def to_tensor(self):
        self.as_tensor = True
        return self

    def to_vector(self):
        if not self.is_batch:
            raise ValueError("Cannot vectorise an unbatched dataset")
        self.is_vector = True
        return self
