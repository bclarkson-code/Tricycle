import numpy as np

from tricycle.einsum import Einsum
from tricycle.tensor import to_tensor
from tricycle.unary import (
    uadd,
    ucos,
    udiv,
    uerf,
    uexp,
    ulog,
    umax,
    umin,
    umul,
    upow,
    usin,
    usqrt,
    usub,
)


def test_can_add():
    in_tensor = to_tensor([0, 1, 2, 3])
    out_tensor = uadd(in_tensor, 1)

    correct = np.array([1, 2, 3, 4])

    assert out_tensor.close_to(correct)

    out_tensor.backward()

    correct = np.ones_like(correct)
    assert in_tensor.grad is not None
    assert in_tensor.grad.close_to(correct)


def test_can_differentiate_einsum():
    left = to_tensor(np.arange(12).reshape(3, 4))
    right = to_tensor(np.arange(12).reshape(4, 3))

    out_tensor = Einsum("ij,jk->ik")(left, right)

    assert out_tensor.close_to(
        [[42, 48, 54], [114, 136, 158], [186, 224, 262]]
    )
    out_tensor.backward()

    assert left.grad is not None
    assert left.grad.close_to(
        [[3, 12, 21, 30], [3, 12, 21, 30], [3, 12, 21, 30]]
    )

    assert right.grad is not None
    assert right.grad.close_to(
        [[12, 12, 12], [15, 15, 15], [18, 18, 18], [21, 21, 21]]
    )


def test_can_umul():
    in_tensor = to_tensor([0, 1, 2, 3])
    out_tensor = umul(in_tensor, 2)

    assert out_tensor.shape == (4,)
    assert out_tensor.close_to(np.array([0, 2, 4, 6]))

    out_tensor.backward()

    assert in_tensor.grad is not None
    assert in_tensor.grad.close_to([2, 2, 2, 2])


def test_can_usub():
    in_tensor = to_tensor([1, 2, 3, 4])

    # subtract 1 from each element
    out_tensor = usub(in_tensor, 1)

    assert out_tensor.shape == (4,)
    assert out_tensor.close_to(np.array([0, 1, 2, 3]))

    out_tensor.backward()
    correct = np.ones(in_tensor.shape)

    assert in_tensor.grad is not None
    assert in_tensor.grad.close_to(correct)


def test_can_upow():
    in_tensor = to_tensor([1, 2, 3, 4])
    out_tensor = upow(in_tensor, 3)

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
        out_tensor = udiv(2, in_tensor)

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
    out_tensor = umax(in_tensor, 2)

    assert out_tensor.shape == (4,)
    assert out_tensor.close_to([2, 2, 3, 4])

    out_tensor.backward()

    correct = [0, 0, 1, 1]

    assert in_tensor.grad is not None
    assert in_tensor.grad.close_to(correct)


def test_can_umin():
    in_tensor = to_tensor([1, 2, 3, 4])
    out_tensor = umin(in_tensor, 3)

    assert out_tensor.shape == (4,)
    assert out_tensor.close_to([1, 2, 3, 3])

    out_tensor.backward()

    correct = [1, 1, 0, 0]

    assert in_tensor.grad is not None
    assert in_tensor.grad.close_to(correct)


def test_can_uexp():
    in_tensor = to_tensor([1, 2, 3, 4])
    out_tensor = uexp(in_tensor)

    assert out_tensor.shape == (4,)

    correct = np.exp([1, 2, 3, 4])
    assert out_tensor.close_to(correct)

    out_tensor.backward()

    correct = np.exp([1, 2, 3, 4])

    assert in_tensor.grad is not None
    assert in_tensor.grad.close_to(correct)


def test_can_ulog():
    in_tensor = to_tensor([1, 2, 3, 4])
    out_tensor = ulog(in_tensor)

    assert out_tensor.shape == (4,)
    assert out_tensor.close_to([0, np.log(2), np.log(3), np.log(4)])

    out_tensor.backward()

    correct = [1, 1 / 2, 1 / 3, 1 / 4]

    assert in_tensor.grad is not None
    assert in_tensor.grad.close_to(correct)


def test_can_usin():
    in_tensor = to_tensor([1, 2, 3, 4])
    out_tensor = usin(in_tensor)

    assert out_tensor.shape == (4,)
    assert out_tensor.close_to([np.sin(1), np.sin(2), np.sin(3), np.sin(4)])

    out_tensor.backward()

    assert in_tensor.grad is not None
    assert in_tensor.grad.close_to(
        [np.cos(1), np.cos(2), np.cos(3), np.cos(4)]
    )


def test_can_ucos():
    in_tensor = to_tensor([1, 2, 3, 4])
    out_tensor = ucos(in_tensor)

    assert out_tensor.shape == (4,)
    assert out_tensor.close_to([np.cos(1), np.cos(2), np.cos(3), np.cos(4)])

    out_tensor.backward()

    correct = [-np.sin(1), -np.sin(2), -np.sin(3), -np.sin(4)]

    assert in_tensor.grad is not None
    assert in_tensor.grad.close_to(correct)


def test_can_usqrt():
    in_tensor = to_tensor([1, 2, 3, 4])
    out_tensor = usqrt(in_tensor)

    assert out_tensor.shape == (4,)
    assert out_tensor.close_to([1, np.sqrt(2), np.sqrt(3), np.sqrt(4)])

    out_tensor.backward()

    correct = [0.5, 0.35355339, 0.28867513, 0.25]

    assert in_tensor.grad is not None
    assert in_tensor.grad.close_to(correct)


def test_can_uerf():
    in_tensor = to_tensor([1, 2, 3, 4])
    out_tensor = uerf(in_tensor)

    assert out_tensor.shape == (4,)
    assert out_tensor.close_to(
        [0.84270079, 0.99532227, 0.99997791, 0.99999998]
    )

    out_tensor.backward()

    correct = [-1.12837917, -1.12837917, -1.12837917, -1.12837917]

    assert in_tensor.grad is not None
    assert in_tensor.grad.close_to(correct)
