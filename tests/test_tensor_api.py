import uuid
from copy import deepcopy

import numpy as np

from tricycle.binary import badd, bdiv, bmul, bsub
from tricycle.ops import Tensor, to_tensor
from tricycle.unary import uadd, udiv, umul, usub


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

    assert np.allclose(2 / tensor_1, udiv(2, tensor_1))

    assert np.allclose(tensor_1 / tensor_2, bdiv(tensor_1, tensor_2))


def test_can_pow_tensors():
    tensor_1 = to_tensor(np.arange(12).reshape(3, 4))

    assert np.allclose(tensor_1**2, pow(tensor_1, 2))


def test_tensors_have_uuid():
    tensor_1 = Tensor([1, 2, 3])
    assert tensor_1.uuid
    assert isinstance(tensor_1.uuid, uuid.UUID)

    tensor_2 = to_tensor([1, 2, 3])
    assert tensor_2.uuid
    assert isinstance(tensor_2.uuid, uuid.UUID)
