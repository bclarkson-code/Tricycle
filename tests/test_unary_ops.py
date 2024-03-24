from copy import copy

import numpy as np

from tricycle.einsum import Einsum
from tricycle.tensor import to_tensor
from tricycle.unary import (
    uadd,
    ucos,
    udiv,
    uexp,
    ulog,
    umax,
    umin,
    umul,
    upow,
    usin,
    usub,
)


def test_can_add():
    in_tensor = to_tensor([0, 1, 2, 3])
    out_tensor = uadd(in_tensor, 1)

    correct = np.array([1, 2, 3, 4])

    assert np.allclose(out_tensor, correct)

    out_tensor.backward()

    correct = np.ones_like(correct)
    assert np.allclose(in_tensor.grad, correct), in_tensor.grad


def test_can_differentiate_einsum():
    left = to_tensor(np.arange(12).reshape(3, 4))
    right = to_tensor(np.arange(12).reshape(4, 3))

    out_tensor = Einsum("ij,jk->ik")(left, right)

    assert out_tensor.shape == (3, 3)
    assert np.allclose(
        out_tensor, np.array([[42, 48, 54], [114, 136, 158], [186, 224, 262]])
    )
    out_tensor.backward()


def test_can_umul():
    in_tensor = to_tensor([0, 1, 2, 3])
    out_tensor = umul(in_tensor, 2)

    assert out_tensor.shape == (4,)
    assert np.allclose(out_tensor, np.array([0, 2, 4, 6]))

    out_tensor.backward()

    assert np.allclose(in_tensor.grad, np.array([2, 2, 2, 2]))


def test_can_usub():
    in_tensor = to_tensor([1, 2, 3, 4])

    # subtract 1 from each element
    out_tensor = usub(in_tensor, 1)

    assert out_tensor.shape == (4,)
    assert np.allclose(out_tensor, np.array([0, 1, 2, 3]))

    out_tensor.backward()
    correct = np.ones_like(in_tensor)

    assert np.allclose(in_tensor.grad, correct)


def test_can_upow():
    in_tensor = to_tensor([1, 2, 3, 4])
    out_tensor = upow(in_tensor, 3)

    assert out_tensor.shape == (4,)
    assert np.allclose(out_tensor, np.array([1, 8, 27, 64]))

    out_tensor.backward()
    correct = np.array([3, 12, 27, 48])

    assert np.allclose(in_tensor.grad, correct)


def test_can_udiv():
    # 2 divided by each element
    in_tensor = to_tensor(np.arange(12, dtype=float).reshape(3, 4))
    with np.errstate(divide="ignore"):
        out_tensor = udiv(2, in_tensor)

    assert out_tensor.shape == (3, 4)
    assert np.allclose(
        out_tensor,
        np.array(
            [
                [np.inf, 2, 1, 2 / 3],
                [2 / 4, 2 / 5, 2 / 6, 2 / 7],
                [2 / 8, 2 / 9, 2 / 10, 2 / 11],
            ]
        ),
    )
    with np.errstate(divide="ignore"):
        out_tensor.backward()
        correct = -np.power(copy(in_tensor), -2) * 2

    assert np.allclose(in_tensor.grad, correct)


def test_can_umax():
    in_tensor = to_tensor([1, 2, 3, 4])
    out_tensor = umax(in_tensor, 2)

    assert out_tensor.shape == (4,)
    assert np.allclose(out_tensor, [2, 2, 3, 4])

    out_tensor.backward()

    correct = [0, 0, 1, 1]
    assert np.allclose(in_tensor.grad, correct)


def test_can_umin():
    in_tensor = to_tensor([1, 2, 3, 4])
    out_tensor = umin(in_tensor, 3)

    assert out_tensor.shape == (4,)
    assert np.allclose(out_tensor, [1, 2, 3, 3])

    out_tensor.backward()

    correct = [1, 1, 0, 0]
    assert np.allclose(in_tensor.grad, correct)


def test_can_uexp():
    in_tensor = to_tensor([1, 2, 3, 4])
    out_tensor = uexp(in_tensor)

    assert out_tensor.shape == (4,)

    correct = np.exp([1, 2, 3, 4])
    assert np.allclose(out_tensor, correct)

    out_tensor.backward()

    correct = np.exp([1, 2, 3, 4])
    assert np.allclose(in_tensor.grad, correct)


def test_can_ulog():
    in_tensor = to_tensor([1, 2, 3, 4])
    out_tensor = ulog(in_tensor)

    assert out_tensor.shape == (4,)
    assert np.allclose(out_tensor, [0, np.log(2), np.log(3), np.log(4)])

    out_tensor.backward()

    correct = [1, 1 / 2, 1 / 3, 1 / 4]
    assert np.allclose(in_tensor.grad, correct)


def test_can_usin():
    in_tensor = to_tensor([1, 2, 3, 4])
    out_tensor = usin(in_tensor)

    assert out_tensor.shape == (4,)
    assert np.allclose(
        out_tensor, np.array([np.sin(1), np.sin(2), np.sin(3), np.sin(4)])
    )

    out_tensor.backward()

    correct = [np.cos(1), np.cos(2), np.cos(3), np.cos(4)]

    assert np.allclose(in_tensor.grad, correct)


def test_can_ucos():
    in_tensor = to_tensor([1, 2, 3, 4])
    out_tensor = ucos(in_tensor)

    assert out_tensor.shape == (4,)
    assert np.allclose(
        out_tensor, np.array([np.cos(1), np.cos(2), np.cos(3), np.cos(4)])
    )

    out_tensor.backward()

    correct = [-np.sin(1), -np.sin(2), -np.sin(3), -np.sin(4)]

    assert np.allclose(in_tensor.grad, correct)
