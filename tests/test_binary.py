import numpy as np

from tricycle.binary import badd, bdiv, bmax, bmin, bmul, bsub
from tricycle.tensor import to_tensor


def test_can_badd():  # sourcery skip: extract-duplicate-method
    in_tensor_1 = to_tensor(np.arange(12).reshape(3, 4))
    in_tensor_2 = to_tensor(np.arange(1, 13).reshape(3, 4))

    out_tensor = badd(in_tensor_1, in_tensor_2)

    assert out_tensor.shape == (3, 4)

    correct = to_tensor([[1, 3, 5, 7], [9, 11, 13, 15], [17, 19, 21, 23]])
    assert out_tensor.close_to(correct)

    out_tensor.backward()

    correct = to_tensor(
        [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]
    )
    assert in_tensor_1.grad is not None
    assert in_tensor_1.grad.close_to(correct)

    correct = to_tensor(
        [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]
    )
    assert in_tensor_2.grad is not None
    assert in_tensor_2.grad.close_to(correct)


def test_can_bsub():  # sourcery skip: extract-duplicate-method
    in_tensor_1 = to_tensor(np.arange(12).reshape(3, 4), is_vector=True)
    in_tensor_2 = to_tensor(np.arange(1, 13).reshape(3, 4), is_vector=True)

    out_tensor = bsub(in_tensor_1, in_tensor_2)

    assert out_tensor.shape == (3, 4)
    correct = to_tensor([[-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1]])
    assert out_tensor.close_to(correct)

    out_tensor.backward()

    correct = to_tensor(
        [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]
    )
    assert in_tensor_1.grad is not None
    assert in_tensor_1.grad.close_to(correct)

    correct = to_tensor(
        [
            [-1.0, -1.0, -1.0, -1.0],
            [-1.0, -1.0, -1.0, -1.0],
            [-1.0, -1.0, -1.0, -1.0],
        ]
    )
    assert in_tensor_2.grad is not None
    assert in_tensor_2.grad.close_to(correct)


def test_can_bmul():
    in_tensor_1 = to_tensor(np.arange(12).reshape(3, 4), is_vector=True)
    in_tensor_2 = to_tensor(np.arange(1, 13).reshape(3, 4), is_vector=True)

    out_tensor = bmul(in_tensor_1, in_tensor_2)

    assert out_tensor.shape == (3, 4)
    correct = to_tensor([[0, 2, 6, 12], [20, 30, 42, 56], [72, 90, 110, 132]])
    assert out_tensor.close_to(correct)

    out_tensor.backward()

    assert in_tensor_1.grad is not None
    assert in_tensor_2.grad is not None
    assert in_tensor_1.grad.close_to(to_tensor(in_tensor_2))
    assert in_tensor_2.grad.close_to(to_tensor(in_tensor_1))


def test_can_bdiv():
    in_tensor_1 = to_tensor(np.arange(12).reshape(3, 4), is_vector=True)
    in_tensor_2 = to_tensor(np.arange(1, 13).reshape(3, 4), is_vector=True)

    out_tensor = bdiv(in_tensor_1, in_tensor_2)

    assert out_tensor.shape == (3, 4)
    correct = to_tensor(
        [
            [0, 1 / 2, 2 / 3, 3 / 4],
            [4 / 5, 5 / 6, 6 / 7, 7 / 8],
            [8 / 9, 9 / 10, 10 / 11, 11 / 12],
        ]
    )

    assert out_tensor.close_to(correct)

    out_tensor.backward()

    assert in_tensor_1.grad is not None
    assert in_tensor_2.grad is not None
    assert in_tensor_1.grad.close_to(to_tensor(1 / in_tensor_2))

    assert in_tensor_2.grad.close_to(
        to_tensor(-in_tensor_1 / (in_tensor_2**2))
    )


def test_can_bmax():
    in_tensor_1 = to_tensor(np.arange(12).reshape(3, 4), is_vector=True)
    in_tensor_2 = to_tensor(
        [[0, 0, 0, 0], [100, 100, 100, 100], [8, 9, 10, 11]], is_vector=True
    )

    out_tensor = bmax(in_tensor_1, in_tensor_2)

    assert out_tensor.shape == (3, 4)
    correct = to_tensor([[0, 1, 2, 3], [100, 100, 100, 100], [8, 9, 10, 11]])
    assert out_tensor.close_to(correct)

    out_tensor.backward()

    one_is_bigger = (in_tensor_1 > in_tensor_2).astype(np.float32)
    two_is_bigger = (in_tensor_1 <= in_tensor_2).astype(np.float32)
    one_is_bigger = to_tensor(one_is_bigger)
    two_is_bigger = to_tensor(two_is_bigger)

    assert in_tensor_1.grad is not None
    assert in_tensor_2.grad is not None
    assert in_tensor_1.grad.close_to(one_is_bigger)
    assert in_tensor_2.grad.close_to(two_is_bigger)


def test_can_bmin():
    in_tensor_1 = to_tensor(np.arange(12).reshape(3, 4), is_vector=True)
    in_tensor_2 = to_tensor(
        [[0, 0, 0, 0], [100, 100, 100, 100], [8, 9, 10, 11]], is_vector=True
    )

    out_tensor = bmin(in_tensor_1, in_tensor_2)

    assert out_tensor.shape == (3, 4)
    correct = to_tensor([[0, 0, 0, 0], [4, 5, 6, 7], [8, 9, 10, 11]])
    assert out_tensor.close_to(correct)

    out_tensor.backward()

    one_is_smaller = (in_tensor_1 < in_tensor_2).astype(np.float32)
    two_is_smaller = (in_tensor_1 >= in_tensor_2).astype(np.float32)
    one_is_smaller = to_tensor(one_is_smaller)
    two_is_smaller = to_tensor(two_is_smaller)

    assert in_tensor_1.grad is not None
    assert in_tensor_2.grad is not None
    assert in_tensor_1.grad.close_to(one_is_smaller)
    assert in_tensor_2.grad.close_to(two_is_smaller)
