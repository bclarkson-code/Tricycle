import numpy as np

from tricycle.tensor import to_tensor
from tricycle.unary import (
    UnaryAdd,
    UnaryCos,
    UnaryDivide,
    UnaryExp,
    UnaryLog,
    UnaryMask,
    UnaryMax,
    UnaryMin,
    UnaryMultiply,
    UnaryPower,
    UnarySin,
    UnarySquareRoot,
    UnarySubtract,
)


def test_can_add():
    in_tensor = to_tensor([0, 1, 2, 3])
    out_tensor = UnaryAdd()(in_tensor, 1)

    correct = np.array([1, 2, 3, 4])

    assert out_tensor.close_to(correct)

    out_tensor.backward()

    correct = np.ones_like(correct)
    assert in_tensor.grad is not None
    assert in_tensor.grad.close_to(correct)


def test_can_umul():
    in_tensor = to_tensor([0, 1, 2, 3])
    out_tensor = UnaryMultiply()(in_tensor, 2)

    assert out_tensor.shape == (4,)
    assert out_tensor.close_to(np.array([0, 2, 4, 6]))

    out_tensor.backward()

    assert in_tensor.grad is not None
    assert in_tensor.grad.close_to([2, 2, 2, 2])


def test_can_usub():
    in_tensor = to_tensor([1, 2, 3, 4])

    # subtract 1 from each element
    out_tensor = UnarySubtract()(in_tensor, 1)

    assert out_tensor.shape == (4,)
    assert out_tensor.close_to(np.array([0, 1, 2, 3]))

    out_tensor.backward()
    correct = np.ones(in_tensor.shape)

    assert in_tensor.grad is not None
    assert in_tensor.grad.close_to(correct)


def test_can_upow():
    in_tensor = to_tensor([1, 2, 3, 4])
    out_tensor = UnaryPower()(in_tensor, 3)

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
        out_tensor = UnaryDivide()(2, in_tensor)

    assert out_tensor.shape == (3, 4)
    assert out_tensor.close_to(
        [
            [np.inf, 2, 1, 2 / 3],
            [2 / 4, 2 / 5, 2 / 6, 2 / 7],
            [2 / 8, 2 / 9, 2 / 10, 2 / 11],
        ],
        rtol=1e-3,
    )
    with np.errstate(divide="ignore"):
        out_tensor.backward()
        correct = -np.power(in_tensor.array, -2) * 2

        assert in_tensor.grad is not None
        assert in_tensor.grad.close_to(correct, rtol=1e-3)


def test_can_umax():
    in_tensor = to_tensor([1, 2, 3, 4])
    out_tensor = UnaryMax()(in_tensor, 2)

    assert out_tensor.shape == (4,)
    assert out_tensor.close_to([2, 2, 3, 4])

    out_tensor.backward()

    correct = [0, 0, 1, 1]

    assert in_tensor.grad is not None
    assert in_tensor.grad.close_to(correct)


def test_can_umin():
    in_tensor = to_tensor([1, 2, 3, 4])
    out_tensor = UnaryMin()(in_tensor, 3)

    assert out_tensor.shape == (4,)
    assert out_tensor.close_to([1, 2, 3, 3])

    out_tensor.backward()

    correct = [1, 1, 0, 0]

    assert in_tensor.grad is not None
    assert in_tensor.grad.close_to(correct)


def test_can_uexp():
    in_tensor = to_tensor([1, 2, 3, 4])
    out_tensor = UnaryExp()(in_tensor)

    assert out_tensor.shape == (4,)

    correct = np.exp([1, 2, 3, 4])
    assert out_tensor.close_to(correct, rtol=1e-3)

    out_tensor.backward()

    correct = np.exp([1, 2, 3, 4])

    assert in_tensor.grad is not None
    assert in_tensor.grad.close_to(correct, rtol=1e-3)


def test_can_ulog():
    in_tensor = to_tensor([1, 2, 3, 4])
    out_tensor = UnaryLog()(in_tensor)

    assert out_tensor.shape == (4,)
    assert out_tensor.close_to([0, np.log(2), np.log(3), np.log(4)], rtol=1e-3)

    out_tensor.backward()

    correct = [1, 1 / 2, 1 / 3, 1 / 4]

    assert in_tensor.grad is not None
    assert in_tensor.grad.close_to(correct, rtol=1e-3)


def test_can_usin():
    in_tensor = to_tensor([1, 2, 3, 4])
    out_tensor = UnarySin()(in_tensor)

    assert out_tensor.shape == (4,)
    assert out_tensor.close_to(
        [np.sin(1), np.sin(2), np.sin(3), np.sin(4)], rtol=1e-3
    )

    out_tensor.backward()

    assert in_tensor.grad is not None
    assert in_tensor.grad.close_to(
        [np.cos(1), np.cos(2), np.cos(3), np.cos(4)], rtol=1e-3
    )


def test_can_ucos():
    in_tensor = to_tensor([1, 2, 3, 4])
    out_tensor = UnaryCos()(in_tensor)

    assert out_tensor.shape == (4,)
    assert out_tensor.close_to(
        [np.cos(1), np.cos(2), np.cos(3), np.cos(4)], rtol=1e-3
    )

    out_tensor.backward()

    correct = [-np.sin(1), -np.sin(2), -np.sin(3), -np.sin(4)]

    assert in_tensor.grad is not None
    assert in_tensor.grad.close_to(correct, rtol=1e-3)


def test_can_usqrt():
    in_tensor = to_tensor([1, 2, 3, 4])
    out_tensor = UnarySquareRoot()(in_tensor)

    assert out_tensor.shape == (4,)
    assert out_tensor.close_to(
        [1, np.sqrt(2), np.sqrt(3), np.sqrt(4)], rtol=1e-3
    )

    out_tensor.backward()

    correct = [0.5, 0.35355339, 0.28867513, 0.25]

    assert in_tensor.grad is not None
    assert in_tensor.grad.close_to(correct, rtol=1e-3)


def test_can_bmask():
    in_tensor = to_tensor(np.arange(12).reshape(3, 4), is_batched=True)
    mask = to_tensor(
        [[0, 0, 0, 0], [1, 0, 1, 0], [1, 1, 1, 1]],
        is_batched=True,
        requires_grad=False,
    )
    out_tensor = UnaryMask()(in_tensor, mask)

    assert out_tensor.shape == (3, 4)
    assert out_tensor.close_to([[0, 0, 0, 0], [4, 0, 6, 0], [8, 9, 10, 11]])

    out_tensor.backward()

    assert mask.grad is None
    assert in_tensor.grad is not None
    assert in_tensor.grad.close_to([[0, 0, 0, 0], [1, 0, 1, 0], [1, 1, 1, 1]])
