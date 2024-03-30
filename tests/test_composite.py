import numpy as np

from tricycle.ops import split
from tricycle.tensor import to_tensor


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


def test_mean():
    in_tensor = to_tensor([1, 2, 3, 4, 5, 6])

    out_tensor = in_tensor.mean()

    assert out_tensor.close_to(3.5)

    out_tensor.backward()

    assert in_tensor.grad is not None
    assert in_tensor.grad.close_to([1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6])


def test_standard_deviation():
    in_tensor = to_tensor([1, 2, 3, 4, 5, 6])

    out_tensor = in_tensor.standard_deviation()

    assert out_tensor.close_to(np.std([1, 2, 3, 4, 5, 6]))

    out_tensor.backward()

    assert in_tensor.grad is not None
    diff = 0.09759
    correct = np.array(
        [
            -3 * diff,
            -2 * diff,
            -1 * diff,
            0,
            diff,
            2 * diff,
        ]
    )
    correct += diff / 2
    assert in_tensor.grad.close_to(correct)
