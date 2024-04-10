import textwrap

import numpy as np

from tricycle.functions import softmax, softmax_v2
from tricycle.tensor import to_tensor


def softmax_original():
    batch_size = 4
    shape = (512, 384)

    inputs = to_tensor(
        np.random.random(size=(batch_size, *shape)),
        requires_grad=True,
    )
    inputs = inputs.to_vector()
    fn = softmax

    for _ in range(10):
        out = fn(inputs)
        out.backward()
        out.zero_grad()


def softmax_new():
    batch_size = 4
    shape = (512, 384)

    inputs = to_tensor(
        np.random.random(size=(batch_size, *shape)),
        requires_grad=True,
    )
    inputs = inputs.to_vector()
    fn = softmax_v2

    for _ in range(10):
        out = fn(inputs)
        out.backward()
        out.zero_grad()


__benchmarks__ = [
    (
        softmax_new,
        softmax_original,
        "Trial of softmax to tune params",
    )
]
