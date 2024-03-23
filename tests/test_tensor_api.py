import uuid
from copy import deepcopy

import numpy as np

from tricycle.binary import badd, bdiv, bmul, bsub
from tricycle.ops import Tensor, to_tensor
from tricycle.unary import uadd, udiv, umul, usub


def test_can_add_tensors():
    tensor_1 = to_tensor(np.arange(12).reshape(3, 4))
    tensor_2 = to_tensor(np.arange(12).reshape(3, 4))

    assert (tensor_1 + 1).close_to(uadd(tensor_1, 1))

    assert (tensor_1 + tensor_2).close_to(badd(tensor_1, tensor_2))

    before = deepcopy(tensor_1)
    tensor_1 += 1

    assert tensor_1.close_to(uadd(before, 1))


def test_can_subtract_tensors():
    tensor_1 = to_tensor(np.arange(12).reshape(3, 4))
    tensor_2 = to_tensor(np.arange(12).reshape(3, 4))

    assert (tensor_1 - 1).close_to(usub(tensor_1, 1))

    assert (tensor_1 - tensor_2).close_to(bsub(tensor_1, tensor_2))

    before = deepcopy(tensor_1)
    tensor_1 -= 1

    assert (tensor_1).close_to(usub(before, 1))


def test_can_multiply_tensors():
    tensor_1 = to_tensor(np.arange(12).reshape(3, 4))
    tensor_2 = to_tensor(np.arange(12).reshape(3, 4))

    assert (tensor_1 * 2).close_to(umul(tensor_1, 2))

    assert (tensor_1 * tensor_2).close_to(bmul(tensor_1, tensor_2))

    before = deepcopy(tensor_1)
    tensor_1 *= 2

    assert (tensor_1).close_to(umul(before, 2))


def test_can_divide_tensors():
    tensor_1 = to_tensor(np.arange(1, 13).reshape(3, 4).astype(float))
    tensor_2 = to_tensor(np.arange(1, 13).reshape(3, 4).astype(float))

    assert (2 / tensor_1).close_to(udiv(2, tensor_1))

    assert (tensor_1 / tensor_2).close_to(bdiv(tensor_1, tensor_2))


def test_can_pow_tensors():
    tensor_1 = to_tensor(np.arange(12).reshape(3, 4))

    assert (tensor_1**2).close_to(pow(tensor_1, 2))


def test_tensors_have_uuid():
    tensor_1 = Tensor([1, 2, 3])
    assert tensor_1.uuid
    assert isinstance(tensor_1.uuid, uuid.UUID)

    tensor_2 = to_tensor([1, 2, 3])
    assert tensor_2.uuid
    assert isinstance(tensor_2.uuid, uuid.UUID)
