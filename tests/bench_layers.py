from abc import ABC, abstractmethod
from string import ascii_letters
from typing import Sequence

import numpy as np
from numba import jit

from tricycle.binary import bmul
from tricycle.einsum import Einsum
from tricycle.initialisers import init_xavier
from tricycle.layers import Embedding, Layer
from tricycle.optimisers import Optimiser
from tricycle.tensor import Tensor, nothing, to_tensor


class NewEmbedding(Layer):
    """
    Convert an index to an embedding with a lookup (rather than a one-hot
    encoding and a matrix multiplication)
    """

    def __init__(self, from_size: int, to_size: int, initialiser=init_xavier):
        self.weights = initialiser((from_size, to_size))
        self.vocab_size = from_size

    def forward(self, tensor: Tensor):
        assert (
            tensor.requires_grad is False
        ), "Cannot embed a differentiable tensor"

        if tensor.is_vector:
            result = tensor.xp.stack(
                [self.weights._data[idx] for idx in tensor._data]
            )
            result = to_tensor(
                result,
                is_vector=True,
            )
        else:
            result = to_tensor(self.weights[tensor._data], is_vector=False)

        result.args = (tensor, self.weights)

        def _embed_back_fn(grad: Tensor):
            xp = grad.xp
            if grad.is_vector:
                out = xp.zeros((grad.shape[0], *self.weights.shape))
            else:
                out = xp.zeros(self.weights.shape)

            if grad.is_vector:
                for batch_idx, (batch, tokens) in enumerate(zip(grad, tensor)):
                    for token_idx, (row, token) in enumerate(
                        zip(batch, tokens)
                    ):
                        out[batch_idx][int(token._data)] += grad[batch_idx][
                            token_idx
                        ]._data
            return out

        result.back_fns = (nothing, _embed_back_fn)
        return result

    def _raise_exception(self, *args):
        """
        I haven't figured out how 2nd order derivatives work yet so we'll
        raise an exception for now
        """
        raise NotImplementedError(
            "2nd order derivatives for embedding are not yet implemented"
        )

    def update(self, optimiser: Optimiser):
        self.weights = optimiser(self.weights)

    def zero_grad(self):
        self.weights.grad = None

    def to_gpu(self):
        self.weights.to_gpu()

    def from_gpu(self):
        self.weights.from_gpu()


def test_embedding_back_new():
    n_tokens = 100
    vocab_size = 1024
    batch_size = 4
    embed_dim = 384

    input = to_tensor(
        np.random.randint(0, vocab_size, (batch_size, n_tokens)),
        requires_grad=False,
        dtype=int,
    )
    input = input.to_vector()
    layer = NewEmbedding(from_size=vocab_size, to_size=embed_dim)

    for _ in range(10):
        out = layer(input)
        out.backward()
        out.zero_grad()


def test_embedding_original():
    n_tokens = 100
    vocab_size = 1024
    batch_size = 4
    embed_dim = 384

    input = to_tensor(
        np.random.randint(0, vocab_size, (batch_size, n_tokens)),
        requires_grad=False,
        dtype=int,
    )
    input = input.to_vector()
    layer = Embedding(from_size=vocab_size, to_size=embed_dim)

    for _ in range(10):
        out = layer(input)
        out.backward()
        out.zero_grad()


#
# __benchmarks__ = [
#     (
#         test_embedding_original,
#         test_embedding_back_new,
#         "Forward pass of embedding layer",
#     )
# ]
