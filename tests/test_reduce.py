import numpy as np

from tricycle.ops import to_tensor
from tricycle.reduce import ReduceMax, ReduceMin


def test_can_rmax():
    in_tensor = to_tensor(np.arange(3 * 4 * 5).reshape(3, 4, 5))

    out_tensor = ReduceMax()(in_tensor, "ijk->ik")

    assert out_tensor.shape == (3, 5)
    assert out_tensor.close_to(
        [[15, 16, 17, 18, 19], [35, 36, 37, 38, 39], [55, 56, 57, 58, 59]]
    )

    out_tensor.backward()
    correct = [
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
        ],
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
        ],
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
        ],
    ]

    assert in_tensor.grad is not None
    assert in_tensor.grad.close_to(correct)


def test_can_rmin():
    in_tensor = to_tensor(np.arange(3 * 4 * 5).reshape(3, 4, 5))

    out_tensor = ReduceMin()(in_tensor, "ijk->ik")

    assert out_tensor.shape == (3, 5)
    assert out_tensor.close_to(
        [[0, 1, 2, 3, 4], [20, 21, 22, 23, 24], [40, 41, 42, 43, 44]]
    )

    out_tensor.backward()

    correct = [
        [
            [1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        [
            [1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        [
            [1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
    ]

    assert in_tensor.grad.close_to(correct)
