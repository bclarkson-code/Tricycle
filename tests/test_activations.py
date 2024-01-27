import numpy as np

from tricycle.activation import ReLU
from tricycle.tensor import to_tensor


def test_relu():
    x = to_tensor([-1, 0, 1])
    relu = ReLU()
    y = relu(x)
    assert np.allclose(y, [0, 0, 1])
