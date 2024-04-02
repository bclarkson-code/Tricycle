import hypothesis.strategies as st
import pytest
from hypothesis import assume, given
from hypothesis.extra.array_api import make_strategies_namespace
from numpy import array_api as xp

from tricycle.binary import _shapes_match
from tricycle.tensor import to_tensor

np_strategies = make_strategies_namespace(xp)


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


# Test the addition operation
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


if __name__ == "__main__":
    pytest.main([__file__])
