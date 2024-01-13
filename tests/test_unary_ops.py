import numpy as np

from tricycle_v2.ops import einsum, to_tensor
from tricycle_v2.unary import uadd, umul, usub


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
    out_tensor = usub(1, in_tensor)

    assert out_tensor.shape == (3, 4)
    assert np.allclose(
        out_tensor, np.array([[1, 0, -1, -2], [-3, -4, -5, -6], [-7, -8, -9, -10]])
    )

    out_tensor.backward()

    assert np.allclose(
        in_tensor.grad, np.array([[-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1]])
    )
