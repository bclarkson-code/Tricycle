import numpy as np

from tricycle.ops import to_tensor
from tricycle.reduce import radd, rmax, rmin


def test_can_radd():
    in_tensor = to_tensor(np.arange(3 * 4 * 5).reshape(3, 4, 5))

    out_tensor = radd(in_tensor, "ijk->ik")

    assert out_tensor.shape == (3, 5)

    assert np.allclose(
        out_tensor,
        np.array(
            [[30, 34, 38, 42, 46], [110, 114, 118, 122, 126], [190, 194, 198, 202, 206]]
        ),
    )

    out_tensor.backward()

    assert np.allclose(
        in_tensor.grad,
        np.ones_like(in_tensor),
    )


def test_can_rmax():
    in_tensor = to_tensor(np.arange(3 * 4 * 5).reshape(3, 4, 5))

    out_tensor = rmax(in_tensor, "ijk->ik")

    assert out_tensor.shape == (3, 5)
    assert np.allclose(
        out_tensor,
        np.array([[15, 16, 17, 18, 19], [35, 36, 37, 38, 39], [55, 56, 57, 58, 59]]),
    )

    out_tensor.backward()

    assert np.allclose(
        in_tensor.grad,
        [
            [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1]],
            [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1]],
            [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1]],
        ],
    )


def test_can_rmin():
    in_tensor = to_tensor(np.arange(3 * 4 * 5).reshape(3, 4, 5))

    out_tensor = rmin(in_tensor, "ijk->ik")

    assert out_tensor.shape == (3, 5)
    assert np.allclose(
        out_tensor,
        np.array([[0, 1, 2, 3, 4], [20, 21, 22, 23, 24], [40, 41, 42, 43, 44]]),
    )

    out_tensor.backward()

    assert np.allclose(
        in_tensor.grad,
        [
            [[1, 1, 1, 1, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            [[1, 1, 1, 1, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            [[1, 1, 1, 1, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        ],
    )
