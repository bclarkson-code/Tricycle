# tests/test_python_and_numpy.py
import numpy as np


def test_can_sum():
    a = np.array([1, 2, 3])
    assert a.sum() == 6


def test_can_do_matrix_algebra():
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    b = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert np.allclose(b @ a, a @ b)
    assert np.allclose(a, a @ b)
