import numpy as np
from matplotlib.pyplot import axis

from tricycle.ops import softmax, split
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


def test_split_first_axis():
    in_tensor = to_tensor([1, 2, 3, 4, 5, 6])

    out_tensors = split(in_tensor, 3)

    assert len(out_tensors) == 3

    assert out_tensors[0].shape == (2,)
    assert out_tensors[1].shape == (2,)
    assert out_tensors[2].shape == (2,)

    assert out_tensors[0].close_to([1, 2])
    assert out_tensors[1].close_to([3, 4])
    assert out_tensors[2].close_to([5, 6])

    out_tensors[0].backward()
    assert in_tensor.grad is not None
    assert in_tensor.grad.close_to([1, 1, 0, 0, 0, 0])

    out_tensors[1].backward()
    assert in_tensor.grad is not None
    assert in_tensor.grad.close_to([1, 1, 1, 1, 0, 0])

    out_tensors[2].backward()
    assert in_tensor.grad is not None
    assert in_tensor.grad.close_to([1, 1, 1, 1, 1, 1])


def test_split_middle_axis():
    in_tensor = to_tensor(np.ones((2, 3, 4)))

    out_tensors = split(in_tensor, n_splits=2, axis=-1)

    assert len(out_tensors) == 2

    assert out_tensors[0].shape == (2, 3, 2)
    assert out_tensors[1].shape == (2, 3, 2)

    assert out_tensors[0].close_to(np.ones((2, 3, 2)))
    assert out_tensors[1].close_to(np.ones((2, 3, 2)))

    out_tensors[0].backward()
    assert in_tensor.grad is not None
    assert in_tensor.grad.close_to([[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0]])


def test_reshape():
    in_tensor = to_tensor([1, 2, 3, 4, 5, 6])

    out_tensor = in_tensor.reshape((2, 3))

    assert out_tensor.shape == (2, 3)
    assert out_tensor.close_to([[1, 2, 3], [4, 5, 6]])

    out_tensor.backward()

    assert in_tensor.grad is not None
    assert in_tensor.grad.close_to([1, 1, 1, 1, 1, 1])
