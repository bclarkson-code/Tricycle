import numpy as np

from tricycle.functions import sigmoid, softmax
from tricycle.tensor import to_tensor


def test_softmax():
    in_tensor = to_tensor([1, 2, 3, 1])

    out_tensor = softmax(in_tensor)

    assert out_tensor.shape == (4,)
    out_tensor.backward()

    # manually figure out softmax derivative using
    # d S(x) / d x = S(x) * (1 - S(x))
    left = np.einsum("i,ij->ij", out_tensor, np.eye(4, 4))
    right = np.einsum("i,j->ij", out_tensor, out_tensor)
    correct = (left - right) @ np.ones_like(in_tensor)

    assert in_tensor.grad is not None
    assert in_tensor.grad.close_to(correct)


def test_softmax_multi_dimension():
    in_tensor = to_tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

    out_tensor = softmax(in_tensor)

    assert out_tensor.shape == (2, 2, 2)
    assert out_tensor.close_to(
        [
            [[0.26894142, 0.73105858], [0.26894142, 0.73105858]],
            [[0.26894142, 0.73105858], [0.26894142, 0.73105858]],
        ],
    )

    out_tensor.backward()


def test_sigmoid():
    in_tensor = to_tensor([0, 1, 2, 3])
    out_tensor = sigmoid(in_tensor)

    assert out_tensor.shape == (4,)
    assert out_tensor.close_to([0.5, 0.73105858, 0.88079708, 0.95257413])

    out_tensor.backward()
    correct_grad = out_tensor * (1 - out_tensor)

    assert in_tensor.grad is not None
    assert in_tensor.grad.close_to(correct_grad)
