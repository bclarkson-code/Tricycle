from copy import deepcopy

import numpy as np

from tricycle_v2.binary import badd, bdiv, bmul, bsub
from tricycle_v2.ops import to_tensor
from tricycle_v2.unary import uadd, udiv, umul, usub


def test_can_add_tensors():
    tensor_1 = to_tensor(np.arange(12).reshape(3, 4))
    tensor_2 = to_tensor(np.arange(12).reshape(3, 4))

    assert np.allclose(tensor_1 + 1, uadd(tensor_1, 1))

    assert np.allclose(tensor_1 + tensor_2, badd(tensor_1, tensor_2))

    before = deepcopy(tensor_1)
    tensor_1 += 1

    assert np.allclose(tensor_1, uadd(before, 1))


def test_can_subtract_tensors():
    tensor_1 = to_tensor(np.arange(12).reshape(3, 4))
    tensor_2 = to_tensor(np.arange(12).reshape(3, 4))

    assert np.allclose(tensor_1 - 1, usub(tensor_1, 1))

    assert np.allclose(tensor_1 - tensor_2, bsub(tensor_1, tensor_2))

    before = deepcopy(tensor_1)
    tensor_1 -= 1

    assert np.allclose(tensor_1, usub(before, 1))


def test_can_multiply_tensors():
    tensor_1 = to_tensor(np.arange(12).reshape(3, 4))
    tensor_2 = to_tensor(np.arange(12).reshape(3, 4))

    assert np.allclose(tensor_1 * 2, umul(tensor_1, 2))

    assert np.allclose(tensor_1 * tensor_2, bmul(tensor_1, tensor_2))

    before = deepcopy(tensor_1)
    tensor_1 *= 2

    assert np.allclose(tensor_1, umul(before, 2))


def test_can_divide_tensors():
    tensor_1 = to_tensor(np.arange(1, 13).reshape(3, 4).astype(float))
    tensor_2 = to_tensor(np.arange(1, 13).reshape(3, 4).astype(float))

    assert np.allclose(tensor_1 / 2, udiv(tensor_1, 2))

    assert np.allclose(tensor_1 / tensor_2, bdiv(tensor_1, tensor_2))

    before = deepcopy(tensor_1)
    tensor_1 /= 2.0

    assert np.allclose(tensor_1, udiv(before, 2))


def test_can_pow_tensors():
    tensor_1 = to_tensor(np.arange(12).reshape(3, 4))

    assert np.allclose(tensor_1**2, pow(tensor_1, 2))
