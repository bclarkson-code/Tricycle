import numpy as np

from tricycle.tensor import to_tensor
from tricycle.unary import (
    UAdd,
    UCos,
    UDiv,
    UExp,
    ULog,
    UMax,
    UMin,
    UMul,
    UPow,
    USin,
    USqrt,
    USub,
)


def test_can_add():
    in_tensor = to_tensor([0, 1, 2, 3])
    out_tensor = UAdd()(in_tensor, 1)

    correct = np.array([1, 2, 3, 4])

    assert out_tensor.close_to(correct)

    out_tensor.backward()

    correct = np.ones_like(correct)
    assert in_tensor.grad is not None
    assert in_tensor.grad.close_to(correct)


def test_can_umul():
    in_tensor = to_tensor([0, 1, 2, 3])
    out_tensor = UMul()(in_tensor, 2)

    assert out_tensor.shape == (4,)
    assert out_tensor.close_to(np.array([0, 2, 4, 6]))

    out_tensor.backward()

    assert in_tensor.grad is not None
    assert in_tensor.grad.close_to([2, 2, 2, 2])


def test_can_usub():
    in_tensor = to_tensor([1, 2, 3, 4])

    # subtract 1 from each element
    out_tensor = USub()(in_tensor, 1)

    assert out_tensor.shape == (4,)
    assert out_tensor.close_to(np.array([0, 1, 2, 3]))

    out_tensor.backward()
    correct = np.ones(in_tensor.shape)

    assert in_tensor.grad is not None
    assert in_tensor.grad.close_to(correct)


def test_can_upow():
    in_tensor = to_tensor([1, 2, 3, 4])
    out_tensor = UPow()(in_tensor, 3)

    assert out_tensor.shape == (4,)
    assert out_tensor.close_to(np.array([1, 8, 27, 64]))

    out_tensor.backward()
    correct = np.array([3, 12, 27, 48])

    assert in_tensor.grad is not None
    assert in_tensor.grad.close_to(correct)


def test_can_udiv():
    # 2 divided by each element
    in_tensor = to_tensor(np.arange(12, dtype=float).reshape(3, 4))
    with np.errstate(divide="ignore"):
        out_tensor = UDiv()(2, in_tensor)

    assert out_tensor.shape == (3, 4)
    assert out_tensor.close_to(
        [
            [np.inf, 2, 1, 2 / 3],
            [2 / 4, 2 / 5, 2 / 6, 2 / 7],
            [2 / 8, 2 / 9, 2 / 10, 2 / 11],
        ]
    )
    with np.errstate(divide="ignore"):
        out_tensor.backward()
        correct = -np.power(in_tensor._data, -2) * 2

        assert in_tensor.grad is not None
        assert in_tensor.grad.close_to(correct)


def test_can_umax():
    in_tensor = to_tensor([1, 2, 3, 4])
    out_tensor = UMax()(in_tensor, 2)

    assert out_tensor.shape == (4,)
    assert out_tensor.close_to([2, 2, 3, 4])

    out_tensor.backward()

    correct = [0, 0, 1, 1]

    assert in_tensor.grad is not None
    assert in_tensor.grad.close_to(correct)


def test_can_umin():
    in_tensor = to_tensor([1, 2, 3, 4])
    out_tensor = UMin()(in_tensor, 3)

    assert out_tensor.shape == (4,)
    assert out_tensor.close_to([1, 2, 3, 3])

    out_tensor.backward()

    correct = [1, 1, 0, 0]

    assert in_tensor.grad is not None
    assert in_tensor.grad.close_to(correct)


def test_can_uexp():
    in_tensor = to_tensor([1, 2, 3, 4])
    out_tensor = UExp()(in_tensor)

    assert out_tensor.shape == (4,)

    correct = np.exp([1, 2, 3, 4])
    assert out_tensor.close_to(correct)

    out_tensor.backward()

    correct = np.exp([1, 2, 3, 4])

    assert in_tensor.grad is not None
    assert in_tensor.grad.close_to(correct)


def test_can_ulog():
    in_tensor = to_tensor([1, 2, 3, 4])
    out_tensor = ULog()(in_tensor)

    assert out_tensor.shape == (4,)
    assert out_tensor.close_to([0, np.log(2), np.log(3), np.log(4)])

    out_tensor.backward()

    correct = [1, 1 / 2, 1 / 3, 1 / 4]

    assert in_tensor.grad is not None
    assert in_tensor.grad.close_to(correct)


def test_can_usin():
    in_tensor = to_tensor([1, 2, 3, 4])
    out_tensor = USin()(in_tensor)

    assert out_tensor.shape == (4,)
    assert out_tensor.close_to([np.sin(1), np.sin(2), np.sin(3), np.sin(4)])

    out_tensor.backward()

    assert in_tensor.grad is not None
    assert in_tensor.grad.close_to(
        [np.cos(1), np.cos(2), np.cos(3), np.cos(4)]
    )


def test_can_ucos():
    in_tensor = to_tensor([1, 2, 3, 4])
    out_tensor = UCos()(in_tensor)

    assert out_tensor.shape == (4,)
    assert out_tensor.close_to([np.cos(1), np.cos(2), np.cos(3), np.cos(4)])

    out_tensor.backward()

    correct = [-np.sin(1), -np.sin(2), -np.sin(3), -np.sin(4)]

    assert in_tensor.grad is not None
    assert in_tensor.grad.close_to(correct)


def test_can_usqrt():
    in_tensor = to_tensor([1, 2, 3, 4])
    out_tensor = USqrt()(in_tensor)

    assert out_tensor.shape == (4,)
    assert out_tensor.close_to([1, np.sqrt(2), np.sqrt(3), np.sqrt(4)])

    out_tensor.backward()

    correct = [0.5, 0.35355339, 0.28867513, 0.25]

    assert in_tensor.grad is not None
    assert in_tensor.grad.close_to(correct)
