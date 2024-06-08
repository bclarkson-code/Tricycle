"""
The hardest part of this project is correctness. To ensure that we are correct
we can use hypothesis to compare tricycle functions with pytorch equivalents
"""

from copy import copy
from warnings import warn

import hypothesis.strategies as st
import numpy as np
import torch
from hypothesis import assume, example, given, settings
from hypothesis.extra import numpy as xp

from tricycle import CUPY_ENABLED
from tricycle.functions import Softmax
from tricycle.layers import Dense, Embedding
from tricycle.loss import CrossEntropy
from tricycle.tensor import to_tensor


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
    return draw(st.integers(min_value=1, max_value=64))


@st.composite
def tokens(draw):
    """
    Tokens are a list of integers. They can be either 1d or 2d and vectorised
    """
    shape = draw(xp.array_shapes(min_dims=1, max_dims=2, max_side=64))
    tokens_ = draw(
        xp.arrays(
            xp.integer_dtypes(),
            shape=shape,
            elements=st.integers(min_value=1, max_value=64),
        )
    )
    return to_tensor(
        tokens_, is_vector=len(shape) == 2, dtype=np.int64, requires_grad=False
    )


@st.composite
def tensor_shape(draw, force_divisible_by_32=True):
    shape = draw(
        st.lists(
            st.integers(min_value=1, max_value=64), min_size=1, max_size=3
        )
    )
    if force_divisible_by_32:
        # make one of the dimensions 32 to keep cuda happy
        idx = np.random.choice(list(range(len(shape))))
        shape[idx] = 32
    return shape


@st.composite
def tensor(draw):
    """
    Generate a single, initial tensor (not as the result of an operation)
    For our model, we need the following tensors:
     - 1d non-vector
     - 2d non-vector
     - 2d vector
     - 3d vector
    """
    shape_ = draw(tensor_shape())
    data = draw(xp.arrays(dtype=np.float64, shape=shape_))
    match len(shape_):
        case 1:
            is_vector = False
        case 2:
            is_vector = draw(st.booleans())
        case 3:
            is_vector = True
    requires_grad = True
    if CUPY_ENABLED:
        on_gpu = draw(st.booleans())
    else:
        warn("CUPY_ENABLED = False so GPU tests have been disabled")
        on_gpu = False

    tensor = to_tensor(
        data,
        is_vector=is_vector,
        requires_grad=requires_grad,
    )
    if on_gpu:
        tensor = tensor.to_gpu()
    return tensor, tensor_shape


@st.composite
def embedding_shape(draw):
    """
    Embeddings are either 2d or 3d and vectorised
    """
    return draw(
        st.lists(
            st.integers(min_value=1, max_value=64), min_size=2, max_size=3
        )
    )


def build_tensor(shape_, is_vector):
    """
    Generate a single, initial tensor (not as the result of an operation)
    For our model, we need the following tensors:
     - 1d non-vector
     - 2d non-vector
     - 2d vector
     - 3d vector
    """
    np.random.seed(0)
    data = np.random.random(shape_).astype(np.float32)
    match len(shape_):
        case 1:
            is_vector = False
        case 2:
            is_vector = is_vector
        case 3:
            is_vector = True
    requires_grad = True

    return to_tensor(
        data,
        is_vector=is_vector,
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
    is_vector = len(shape) in {3, 4}
    requires_grad = draw(st.booleans())
    if CUPY_ENABLED:
        on_gpu = draw(st.booleans())
    else:
        warn("CUPY_ENABLED = False so GPU tests have been disabled")
        on_gpu = False

    tensor = to_tensor(
        data,
        is_vector=is_vector,
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
        is_vector = draw(st.booleans())

        if draw(st.booleans()):
            data = data[1:]

        tensor = to_tensor(data, is_vector=is_vector)
        tensors.append(tensor)

    return tensors


@given(tensor_shape(), integer(), st.booleans())
@settings(deadline=1000)
def test_tricycle_dense_matches_pytorch(in_shape, out_shape, is_vector):
    tensor = build_tensor(in_shape, is_vector)
    assume(np.isfinite(tensor._data).all())

    from_size = tensor.shape[-1]

    pt_layer = torch.nn.Linear(
        in_features=from_size, out_features=out_shape, bias=False
    )
    tr_layer = Dense(from_size=from_size, to_size=out_shape)
    tr_layer.weights = to_tensor(pt_layer.weight.detach().numpy().T)

    pt_out = pt_layer(torch.tensor(tensor._data))
    tr_out = tr_layer(tensor)

    assert np.allclose(
        pt_out.detach().numpy(), tr_out.numpy(), rtol=1e-5, atol=1e-5
    )

    pt_out.sum().backward()
    tr_out.from_vector().sum().backward()

    assert np.allclose(
        pt_layer.weight.grad.detach().numpy().T,
        tr_layer.weights.grad.numpy(),
        rtol=1e-4,
    )


@given(tokens(), integer())
@settings(deadline=1000)
# @example(
#     tokens_=to_tensor(
#         [[1, 1], [1, 1]], dtype=np.int64, requires_grad=False, is_vector=True
#     ),
#     out_shape=1,
# )
def test_embedding_matches(tokens_, out_shape):
    vocab_size = tokens_._data.max() + 1
    pt_layer = torch.nn.Embedding(
        num_embeddings=vocab_size, embedding_dim=out_shape
    )
    tr_layer = Embedding(from_size=vocab_size, to_size=out_shape)
    tr_layer.weights = to_tensor(pt_layer.weight.detach().numpy())

    pt_out = pt_layer(torch.tensor(tokens_._data))
    tr_out = tr_layer(tokens_)

    assert np.allclose(
        pt_out.detach().numpy(), tr_out.numpy(), rtol=1e-5, atol=1e-5
    )

    pt_out.sum().backward()
    tr_out.from_vector().sum().backward()

    assert np.allclose(
        pt_layer.weight.grad.detach().numpy(),
        tr_layer.weights.grad.numpy(),
        rtol=1e-4,
    )


@given(tensor_shape(force_divisible_by_32=True), st.booleans())
@settings(deadline=1000)
@example(in_shape=[1, 1, 1, 128], is_vector=True)
def test_tricycle_softmax_matches_pytorch(in_shape, is_vector):
    tensor = build_tensor(in_shape, is_vector)
    assume(np.isfinite(tensor._data).all())

    tensor.requires_grad = True

    pt_layer = torch.nn.functional.softmax
    tr_layer = Softmax()

    pt_input = torch.tensor(tensor._data, requires_grad=True)

    pt_out = pt_layer(pt_input, dim=-1)
    tr_out = tr_layer(tensor)

    assert np.allclose(
        pt_out.detach().numpy(), tr_out.numpy(), rtol=1e-5, atol=1e-5
    )

    pt_out.sum().backward()
    tr_out.from_vector().sum().backward()

    assert np.allclose(
        pt_input.grad.detach().numpy(),
        tensor.grad.numpy(),
        rtol=1e-4,
        atol=1e-5,
    )


@given(tensor_shape(), st.booleans())
@example(in_shape=[2, 2, 4], is_vector=False)
def test_crossentropy_matches(in_shape, is_vector):
    y_pred = build_tensor(in_shape, is_vector)
    y_true = np.random.randint(0, in_shape[-1], size=in_shape[:-1])
    y_true = to_tensor(y_true, is_vector=is_vector, dtype=int)
    assume(np.isfinite(y_pred._data).all())

    tr_out = CrossEntropy()(y_true, y_pred).from_vector()
    if len(in_shape) > 1:
        tr_out = tr_out.mean()

    if len(in_shape) == 1:
        p_y_pred = copy(y_pred._data)
    if len(in_shape) == 2:
        p_y_pred = copy(y_pred._data)
    if len(in_shape) == 3:
        p_y_pred = copy(y_pred._data).transpose(0, -1, 1)
    p_y_pred = torch.tensor(p_y_pred, requires_grad=True)
    p_y_true = torch.tensor(y_true._data, dtype=torch.long)

    p_out = torch.nn.CrossEntropyLoss()(
        input=p_y_pred,
        target=p_y_true,
    )

    assert tr_out.close_to(p_out.detach().numpy().item())

    tr_out.backward()
    p_out.backward()

    p_grad = p_y_pred.grad.detach().numpy()
    if len(in_shape) == 3:
        p_grad = p_grad.transpose(0, -1, 1)
    assert y_pred.grad.close_to(p_grad)
