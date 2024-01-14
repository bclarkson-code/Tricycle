import numpy as np

from tricycle_v2.ops import einsum, to_tensor
from tricycle_v2.unary import uadd, udiv, umax, umul, upow, usub, umin


def test_can_add():
    in_tensor = to_tensor(np.arange(12).reshape(3, 4))
    out_tensor = uadd(in_tensor, 1)

    correct = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

    assert np.allclose(out_tensor, correct)

    out_tensor.backward()

    correct = np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])
    assert np.allclose(in_tensor.grad, correct)


def test_can_differentiate_einsum():
    left = to_tensor(np.arange(12).reshape(3, 4))
    right = to_tensor(np.arange(12).reshape(4, 3))

    out_tensor = einsum("ij,jk->ik", left, right)

    assert out_tensor.shape == (3, 3)
    assert np.allclose(
        out_tensor, np.array([[42, 48, 54], [114, 136, 158], [186, 224, 262]])
    )
    out_tensor.backward()


def test_can_umul():
    in_tensor = to_tensor(np.arange(12).reshape(3, 4))
    out_tensor = umul(in_tensor, 2)

    assert out_tensor.shape == (3, 4)
    assert np.allclose(
        out_tensor, np.array([[0, 2, 4, 6], [8, 10, 12, 14], [16, 18, 20, 22]])
    )

    out_tensor.backward()

    assert np.allclose(
        in_tensor.grad, np.array([[2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2]])
    )


def test_can_usub():
    in_tensor = to_tensor(np.arange(12).reshape(3, 4))

    # subtract 1 from each element
    out_tensor = usub(in_tensor, 1)

    assert out_tensor.shape == (3, 4)
    assert np.allclose(
        out_tensor, np.array([[-1, 0, 1, 2], [3, 4, 5, 6], [7, 8, 9, 10]])
    )

    out_tensor.backward()

    assert np.allclose(
        in_tensor.grad, np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])
    )

    # one subtract each element
    in_tensor = to_tensor(np.arange(12).reshape(3, 4))
    out_tensor = usub(1, in_tensor)

    assert out_tensor.shape == (3, 4)
    assert np.allclose(
        out_tensor, np.array([[1, 0, -1, -2], [-3, -4, -5, -6], [-7, -8, -9, -10]])
    )

    out_tensor.backward()

    assert np.allclose(
        in_tensor.grad, np.array([[-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1]])
    )


def test_can_upow():
    in_tensor = to_tensor(np.arange(12).reshape(3, 4))
    out_tensor = upow(in_tensor, 2)

    assert out_tensor.shape == (3, 4)
    assert np.allclose(
        out_tensor, np.array([[0, 1, 4, 9], [16, 25, 36, 49], [64, 81, 100, 121]])
    )

    out_tensor.backward()

    assert np.allclose(
        in_tensor.grad, np.array([[0, 2, 4, 6], [8, 10, 12, 14], [16, 18, 20, 22]])
    )


def test_can_udiv():
    # each element divided by 2
    in_tensor = to_tensor(np.arange(12).reshape(3, 4))
    out_tensor = udiv(in_tensor, 2)

    assert out_tensor.shape == (3, 4)
    assert np.allclose(
        out_tensor,
        np.array([[0.0, 0.5, 1.0, 1.5], [2.0, 2.5, 3.0, 3.5], [4.0, 4.5, 5.0, 5.5]]),
    )

    out_tensor.backward()

    assert np.allclose(
        in_tensor.grad,
        np.array([[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]]),
    )

    # 2 divided by each element
    in_tensor = to_tensor(np.arange(12).reshape(3, 4))
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


def test_can_umax():
    in_tensor = to_tensor(np.arange(12).reshape(3, 4))
    out_tensor = umax(in_tensor, 4)

    assert out_tensor.shape == (3, 4)
    assert np.allclose(
        out_tensor, np.array([[4, 4, 4, 4], [4, 5, 6, 7], [8, 9, 10, 11]])
    )

    out_tensor.backward()

    assert np.allclose(
        in_tensor.grad,
        np.array([[0, 0, 0, 0], [0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]),
    )


def test_can_umin():
    in_tensor = to_tensor(np.arange(12).reshape(3, 4))
    out_tensor = umin(in_tensor, 4)

    assert out_tensor.shape == (3, 4)
    assert np.allclose(
        out_tensor, np.array([[0, 1, 2, 3], [4, 4, 4, 4], [4, 4, 4, 4]])
    )

    out_tensor.backward()

    assert np.allclose(
        in_tensor.grad,
        np.array([[1.0, 1.0, 1.0, 1.0], [1.0, 0, 0, 0], [0, 0, 0, 0]]),
    )
