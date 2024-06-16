"""
A lot of features of Tricycle are quite hard to test. For example, there are
a lot of different ways to combine `Op`s.

To fix this, this file contains several property-based tests where, instead
of defining a specific situation and checking the output, inputs are generated
randomly and then the outputs are checked against some predefined properties.

For example, in our tokeniser, we want decode(encode(<some_text>)) to return
the input so we can build a property based test that tries a whole bunch
of random inputs and checks that they are unmodified by the decode(encode())
operation.
"""

import numbers
from warnings import warn

import hypothesis.strategies as st
import numpy as np
import pytest
import torch
from hypothesis import assume, given, settings
from hypothesis.extra import numpy as xp

from tricycle import CUPY_ENABLED
from tricycle.binary import (
    BinaryAdd,
    BinaryDivide,
    BinaryMax,
    BinaryMin,
    BinaryMultiply,
    BinarySubtract,
)
from tricycle.einsum import EinsumBackOp
from tricycle.layers import Dense
from tricycle.tensor import to_tensor
from tricycle.tokeniser import BPETokeniser
from tricycle.unary import (
    UnaryAdd,
    UnaryCos,
    UnaryDivide,
    UnaryExp,
    UnaryLog,
    UnaryMax,
    UnaryMin,
    UnaryMultiply,
    UnaryPower,
    UnarySin,
    UnarySquareRoot,
    UnarySubtract,
    nothing,
)
from tricycle.utils import shapes_match


@st.composite
def scalar(draw):
    """
    Generate a single, initial scalar
    """
    group = draw(st.sampled_from(["int", "float", "complex"]))
    if group == "int":
        return draw(st.integers())
    if group == "float":
        return draw(st.floats())
    if group == "complex":
        return draw(st.complex_numbers())


@st.composite
def string(draw):
    return draw(st.text())


@st.composite
def integer(draw):
    return draw(st.integers(min_value=1, max_value=1024))


@st.composite
def unary_op(draw):
    """
    Generate a single, initial unary operation
    """
    ops = [
        UnarySin(),
        UnaryCos(),
        UnaryExp(),
        UnaryLog(),
        UnarySquareRoot(),
    ]
    needs_constant = [
        UnaryAdd(),
        UnarySubtract(),
        UnaryMultiply(),
        UnaryPower(),
        UnaryDivide(),
        UnaryMax(),
        UnaryMin(),
    ]
    op = draw(st.sampled_from(ops))
    if op in needs_constant:
        constant = draw(scalar())
        return op, constant
    return op, None


@st.composite
def binary_op(draw):
    """
    Generate a single, initial binary operation
    """
    ops = [
        BinaryAdd(),
        BinaryDivide(),
        BinaryMax(),
        BinaryMin(),
        BinaryMultiply(),
        BinarySubtract(),
    ]
    return draw(st.sampled_from(ops))


@st.composite
def tensor(draw):
    """
    Generate a single, initial tensor (not as the result of an operation)
    """
    shape = draw(
        xp.array_shapes(min_dims=1, max_dims=4, min_side=1, max_side=32)
    )
    data = draw(xp.arrays(dtype=np.float32, shape=shape))
    match len(shape):
        case 1:
            is_batched = False
        case 2:
            is_batched = draw(st.booleans())
        case 3:
            is_batched = draw(st.booleans())
        case 4:
            is_batched = True
    requires_grad = draw(st.booleans())
    return to_tensor(
        data,
        is_batched=is_batched,
        requires_grad=requires_grad,
    )


@st.composite
def small_tensor(draw):
    """
    Generate a single, initial tensor (not as the result of an operation).
    The tensor can be 1, 2 or 3d
    """
    shape = draw(st.integers(min_value=1, max_value=4))
    data = draw(xp.arrays(dtype=np.float64, shape=shape))
    is_batched = len(shape) in {3, 4}
    requires_grad = draw(st.booleans())
    if CUPY_ENABLED:
        on_gpu = draw(st.booleans())
    else:
        warn("CUPY_ENABLED = False so GPU tests have been disabled")
        on_gpu = False

    tensor = to_tensor(
        data,
        is_batched=is_batched,
        requires_grad=requires_grad,
    )
    if on_gpu:
        tensor = tensor.to_gpu()
    return tensor


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
        data = draw(xp.arrays(dtype=np.float64, shape=shape))
        is_batched = draw(st.booleans())

        if draw(st.booleans()):
            data = data[1:]

        tensor = to_tensor(data, is_batched=is_batched)
        tensors.append(tensor)

    return tensors


@given(tensor_pair_same_shape())
def test_tensor_addition_same_shape(tensors):
    # sourcery skip: no-conditionals-in-tests
    tensor_1, tensor_2 = tensors

    try:
        _shapes_match = shapes_match(tensor_1, tensor_2)
    except ValueError:
        _shapes_match = False

    # sourcery skip: no-conditionals-in-tests
    if not _shapes_match:
        assume(False)

    result = tensor_1 + tensor_2
    largest_input_shape = max(tensor_1.shape, tensor_2.shape)
    assert result.shape == largest_input_shape

    assert result.args == (tensor_1, tensor_2)
    assert result.back_fns == (nothing, nothing)

    assert result.is_batched == tensor_1.is_batched or tensor_2.is_batched


@given(tensor(), scalar())
def test_tensor_addition_scalar(tensor, scalar):
    assume(isinstance(scalar, numbers.Number))
    try:
        assume(abs(scalar) < 2**64)
    except OverflowError:
        assume(False)
    assume(not isinstance(scalar, np.datetime64))
    assume(not isinstance(scalar, np.timedelta64))

    result = tensor + scalar
    assert result.shape == tensor.shape

    assert result.args == (tensor,)
    assert result.back_fns == (nothing,)

    assert result.is_batched == tensor.is_batched


@given(tensor_pair_same_shape())
def test_tensor_multiplication(tensors):
    # sourcery skip: no-conditionals-in-tests
    tensor_1, tensor_2 = tensors

    try:
        _shapes_match = shapes_match(tensor_1, tensor_2)
    except ValueError:
        _shapes_match = False

    # sourcery skip: no-conditionals-in-tests
    if not _shapes_match:
        assume(False)

    result = tensor_1 * tensor_2
    largest_input_shape = max(tensor_1.shape, tensor_2.shape)
    assert result.shape == largest_input_shape

    assert result.args == (tensor_1, tensor_2)
    assert len(result.back_fns) == 2

    assert isinstance(result.back_fns[0], EinsumBackOp)
    assert isinstance(result.back_fns[1], EinsumBackOp)

    assert result.is_batched == tensor_1.is_batched or tensor_2.is_batched


@given(tensor())
def test_close_to(tensor):
    equal_nan = np.isnan(tensor.array).any()

    assert tensor.close_to(tensor, equal_nan=equal_nan, rtol=1e-6, atol=1e-8)


@given(tensor())
def test_can_batch_and_unbatch(tensor):
    assume(not tensor.is_batched)

    batched = tensor.to_batched()
    assert batched.is_batched

    unbatched = batched.from_batched()
    assert not unbatched.is_batched

    assert tensor.close_to(unbatched, equal_nan=True)

    # sourcery skip: no-conditionals-in-tests
    if tensor.requires_grad:
        assert len(unbatched.args) == 1
        assert unbatched.args[0].close_to(tensor, equal_nan=True)

        assert len(unbatched.args[0].args) == 1
        assert unbatched.args[0].args[0].close_to(tensor, equal_nan=True)

        assert unbatched.requires_grad


@given(tensor())
def test_can_move_to_and_from_gpu(tensor):
    # only run this test if we have a gpu enabled
    if not CUPY_ENABLED:
        pytest.skip("GPU not enabled")
    assume(not tensor.on_gpu)

    gpu_tensor = tensor.to_gpu()
    assert gpu_tensor.on_gpu

    cpu_tensor = gpu_tensor.from_gpu()
    assert not cpu_tensor.on_gpu


@given(tensor(), unary_op())
def test_unary_ops(tensor, op):
    # sourcery skip: no-conditionals-in-tests
    op, constant = op
    if constant is not None:
        try:
            assume(abs(constant) < 2**64)
        except OverflowError:
            assume(False)
        result = op(tensor=tensor, constant=constant)
    else:
        result = op(tensor)
    assert result.shape == tensor.shape
    assert result.is_batched == tensor.is_batched
    assert result.on_gpu == tensor.on_gpu


@given(tensor_pair_same_shape(), binary_op())
def test_binary_ops(tensors, op):
    # sourcery skip: no-conditionals-in-tests
    tensor_1, tensor_2 = tensors

    try:
        _shapes_match = shapes_match(tensor_1, tensor_2)
    except ValueError:
        _shapes_match = False
    assume(_shapes_match)

    result = op(tensor_1, tensor_2)

    assert result.shape in [tensor_1.shape, tensor_2.shape]
    assert result.is_batched == any([tensor_1.is_batched, tensor_2.is_batched])
    assert result.on_gpu == any([tensor_1.on_gpu, tensor_2.on_gpu])


@given(string())
def test_tokeniser_encode_decode(text):
    tokeniser = BPETokeniser(vocab_size=1024)
    tokens = tokeniser.encode(text)
    decoded = tokeniser.decode(tokens)

    assert text == decoded


@given(string())
@settings(deadline=1000)
def test_tokeniser_train_encode_decode(text):
    tokeniser = BPETokeniser(vocab_size=1024)

    tokeniser.train(text)

    encoded = tokeniser.encode(text)
    assert np.allclose(encoded, tokeniser.tokens)

    decoded = tokeniser.decode(encoded)
    assert text == decoded
