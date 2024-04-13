import textwrap

import numpy as np

from tricycle.functions import softmax, softmax_v2, softmax_v3, softmax_v4
from tricycle.tensor import to_tensor


def softmax_original():
    batch_size = 16
    shape = (512, 384)

    inputs = to_tensor(
        np.random.random(size=(batch_size, *shape)),
        requires_grad=True,
    ).to_gpu()
    inputs = inputs.to_vector()
    fn = softmax

    for _ in range(10):
        out = fn(inputs)
        out.backward()
        out.cleanup()


def softmax_new():
    batch_size = 16
    shape = (512, 384)

    inputs = to_tensor(
        np.random.random(size=(batch_size, *shape)),
        requires_grad=True,
    ).to_gpu()
    inputs = inputs.to_vector()
    fn = softmax_v2

    for _ in range(10):
        out = fn(inputs)
        out.backward()
        out.cleanup()


def softmax_correct():
    batch_size = 16
    shape = (512, 384)

    inputs = to_tensor(
        np.random.random(size=(batch_size, *shape)),
        requires_grad=True,
    ).to_gpu()
    inputs = inputs.to_vector()
    fn = softmax_v3

    for _ in range(10):
        out = fn(inputs)
        out.backward()
        out.cleanup()


def softmax_cuda():
    batch_size = 16
    shape = (512, 384)

    inputs = to_tensor(
        np.random.random(size=(batch_size, *shape)),
        requires_grad=True,
    ).to_gpu()
    inputs = inputs.to_vector()
    fn = softmax_v4

    for _ in range(10):
        out = fn(inputs)
        out.backward()
        out.cleanup()


__benchmarks__ = [
    # (
    #     softmax_original,
    #     softmax_original,
    #     "Trial of softmax to tune params",
    # ),
    # (
    #     softmax_original,
    #     softmax_new,
    #     "Replace forward pass with scipy softmax",
    # ),
    # (
    #     softmax_original,
    #     softmax_cuda,
    #     "Custom cuda kernel",
    # ),
]
