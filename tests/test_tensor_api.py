from copy import deepcopy

import numpy as np

from tricycle.binary import BAdd, BDiv, BMul, BSub
from tricycle.ops import to_tensor
from tricycle.unary import UAdd, UDiv, UMul, USub


def test_can_add_tensors():
    tensor_1 = to_tensor(np.arange(12).reshape(3, 4))
    tensor_2 = to_tensor(np.arange(12).reshape(3, 4))

    assert (tensor_1 + 1).close_to(UAdd()(tensor_1, 1))

    assert (tensor_1 + tensor_2).close_to(BAdd()(tensor_1, tensor_2))

    before = deepcopy(tensor_1)
    tensor_1 += 1

    assert tensor_1.close_to(UAdd()(before, 1))


def test_can_subtract_tensors():
    tensor_1 = to_tensor(np.arange(12).reshape(3, 4))
    tensor_2 = to_tensor(np.arange(12).reshape(3, 4))

    assert (tensor_1 - 1).close_to(USub()(tensor_1, 1))

    assert (tensor_1 - tensor_2).close_to(BSub()(tensor_1, tensor_2))

    before = deepcopy(tensor_1)
    tensor_1 -= 1

    assert (tensor_1).close_to(USub()(before, 1))


def test_can_multiply_tensors():
    tensor_1 = to_tensor(np.arange(12).reshape(3, 4))
    tensor_2 = to_tensor(np.arange(12).reshape(3, 4))

    assert (tensor_1 * 2).close_to(UMul()(tensor_1, 2))

    assert (tensor_1 * tensor_2).close_to(BMul()(tensor_1, tensor_2))

    before = deepcopy(tensor_1)
    tensor_1 *= 2

    assert (tensor_1).close_to(UMul()(before, 2))


def test_can_divide_tensors():
    tensor_1 = to_tensor(np.arange(1, 13).reshape(3, 4).astype(float))
    tensor_2 = to_tensor(np.arange(1, 13).reshape(3, 4).astype(float))

    assert (2 / tensor_1).close_to(UDiv()(2, tensor_1))

    assert (tensor_1 / tensor_2).close_to(BDiv()(tensor_1, tensor_2))


def test_can_pow_tensors():
    tensor_1 = to_tensor(np.arange(12).reshape(3, 4))

    assert (tensor_1**2).close_to(pow(tensor_1, 2))
