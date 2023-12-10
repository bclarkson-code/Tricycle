import math

import numpy as np

from llm_from_scratch.tensor import tensor


def test_can_handle_scalar():
    tensor(1)


def test_can_handle_vector():
    tensor([1, 2, 3])


def test_can_handle_matrix():
    tensor([[1, 2, 3], [4, 5, 6]])


def test_can_handle_3d():
    tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])


def test_can_handle_irrational_matrix():
    tensor([[1, 2, math.sqrt(3)], [4, 5, 6], [7, 8, 9]])


def test_can_handle_16d():
    # we arent going all the way to 32d because the
    # array ends up being 32GB!
    shape = [2] * 16
    big_array = np.ones(shape)
    tensor(big_array)
