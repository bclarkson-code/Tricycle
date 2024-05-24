from copy import deepcopy

import numpy as np

from tricycle.binary import (
    BinaryAdd,
    BinaryDivide,
    BinaryMultiply,
    BinarySubtract,
)
from tricycle.ops import to_tensor
from tricycle.unary import UnaryAdd, UnaryDivide, UnaryMultiply, UnarySubtract


def test_can_add_tensors():
    tensor_1 = to_tensor(np.arange(12).reshape(3, 4))
    tensor_2 = to_tensor(np.arange(12).reshape(3, 4))

    assert (tensor_1 + 1).close_to(UnaryAdd()(tensor_1, 1))

    assert (tensor_1 + tensor_2).close_to(BinaryAdd()(tensor_1, tensor_2))

    before = deepcopy(tensor_1)
    tensor_1 += 1

    assert tensor_1.close_to(UnaryAdd()(before, 1))


def test_can_subtract_tensors():
    tensor_1 = to_tensor(np.arange(12).reshape(3, 4))
    tensor_2 = to_tensor(np.arange(12).reshape(3, 4))

    assert (tensor_1 - 1).close_to(UnarySubtract()(tensor_1, 1))

    assert (tensor_1 - tensor_2).close_to(BinarySubtract()(tensor_1, tensor_2))

    before = deepcopy(tensor_1)
    tensor_1 -= 1

    assert (tensor_1).close_to(UnarySubtract()(before, 1))


def test_can_multiply_tensors():
    tensor_1 = to_tensor(np.arange(12).reshape(3, 4))
    tensor_2 = to_tensor(np.arange(12).reshape(3, 4))

    assert (tensor_1 * 2).close_to(UnaryMultiply()(tensor_1, 2))

    assert (tensor_1 * tensor_2).close_to(BinaryMultiply()(tensor_1, tensor_2))

    before = deepcopy(tensor_1)
    tensor_1 *= 2

    assert (tensor_1).close_to(UnaryMultiply()(before, 2))


def test_can_divide_tensors():
    tensor_1 = to_tensor(np.arange(1, 13).reshape(3, 4).astype(float))
    tensor_2 = to_tensor(np.arange(1, 13).reshape(3, 4).astype(float))

    assert (2 / tensor_1).close_to(UnaryDivide()(2, tensor_1))

    assert (tensor_1 / tensor_2).close_to(BinaryDivide()(tensor_1, tensor_2))


def test_can_pow_tensors():
    tensor_1 = to_tensor(np.arange(12).reshape(3, 4))

    assert (tensor_1**2).close_to(pow(tensor_1, 2))
