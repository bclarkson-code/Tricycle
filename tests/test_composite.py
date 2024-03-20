import numpy as np

from tricycle.ops import softmax
from tricycle.tensor import to_tensor


def test_softmax():
    in_tensor = to_tensor([1, 2, 3, 1])

    out_tensor = softmax(in_tensor)

    assert out_tensor.shape == (4,)
    out_tensor.backward()

    left = np.einsum("i,ij->ij", out_tensor, np.eye(4, 4))
    right = np.einsum("i,j->ij", out_tensor, out_tensor)
    correct = (left - right) @ np.ones_like(in_tensor)

    assert in_tensor.grad.close_to(correct)
