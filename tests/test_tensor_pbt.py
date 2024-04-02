from copy import copy

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis.extra.array_api import make_strategies_namespace
from numpy import array_api as xp

from tricycle.binary import _shapes_match
from tricycle.einsum import EinsumBackOp
from tricycle.tensor import nothing, to_tensor, unvectorise, vectorise

np_strategies = make_strategies_namespace(xp)


@st.composite
def tensor(draw):
    """
    Generate a single, initial tensor (not as the result of an operation)
    """
    shape = draw(st.integers(min_value=1, max_value=10))
    data = draw(np_strategies.arrays(dtype=xp.float64, shape=shape))
    is_vector = draw(st.booleans())
    requires_grad = draw(st.booleans())

    return to_tensor(
        data,
        is_vector=is_vector,
        requires_grad=requires_grad,
    )


@st.composite
def tensor_pair_same_shape(draw):
    """
    Generate two tensors with the same shape
    """
    shape = draw(st.integers(min_value=1, max_value=10))
    if isinstance(shape, int):
        shape = (shape,)

    tensors = []
    for _ in range(2):
        data = draw(np_strategies.arrays(dtype=xp.float64, shape=shape))
        is_vector = draw(st.booleans())

        if draw(st.booleans()):
            data = data[1:]

        tensor = to_tensor(data, is_vector=is_vector)
        tensors.append(tensor)

    return tensors


@given(tensor_pair_same_shape())
def test_tensor_addition(tensors):
    # sourcery skip: no-conditionals-in-tests
    tensor_1, tensor_2 = tensors

    try:
        shapes_match = _shapes_match(tensor_1, tensor_2)
    except ValueError:
        shapes_match = False

    # sourcery skip: no-conditionals-in-tests
    if not shapes_match:
        assume(False)

    result = tensor_1 + tensor_2
    largest_input_shape = max(tensor_1.shape, tensor_2.shape)
    assert result.shape == largest_input_shape

    assert result.args == (tensor_1, tensor_2)
    assert result.back_fns == (nothing, nothing)

    assert result.is_vector == tensor_1.is_vector or tensor_2.is_vector


@given(tensor_pair_same_shape())
def test_tensor_multiplication(tensors):
    # sourcery skip: no-conditionals-in-tests
    tensor_1, tensor_2 = tensors

    try:
        shapes_match = _shapes_match(tensor_1, tensor_2)
    except ValueError:
        shapes_match = False

    # sourcery skip: no-conditionals-in-tests
    if not shapes_match:
        assume(False)

    result = tensor_1 * tensor_2
    largest_input_shape = max(tensor_1.shape, tensor_2.shape)
    assert result.shape == largest_input_shape

    assert result.args == (tensor_1, tensor_2)
    assert len(result.back_fns) == 2

    assert isinstance(result.back_fns[0], EinsumBackOp)
    assert isinstance(result.back_fns[1], EinsumBackOp)

    assert result.is_vector == tensor_1.is_vector or tensor_2.is_vector


@given(tensor())
def test_close_to(tensor):
    equal_nan = np.isnan(tensor).any()

    assert tensor.close_to(tensor, equal_nan=equal_nan)


@given(tensor())
def test_can_vectorise_and_unvectorise(tensor):
    assume(not tensor.is_vector)

    vectorised = tensor.to_vector()
    assert vectorised.is_vector

    unvectorised = vectorised.from_vector()
    assert not unvectorised.is_vector

    assert tensor.close_to(unvectorised, equal_nan=True)

    # sourcery skip: no-conditionals-in-tests
    if tensor.requires_grad:
        assert len(unvectorised.args) == 1
        assert unvectorised.args[0].close_to(tensor, equal_nan=True)
        assert unvectorised.back_fns == (vectorise,)

        assert len(unvectorised.args[0].args) == 1
        assert (
            unvectorised.args[0].args[0].close_to(tensor, equal_nan=True)
        )
        assert unvectorised.args[0].back_fns == (unvectorise,)

        assert unvectorised.requires_grad


if __name__ == "__main__":
    pytest.main([__file__])
