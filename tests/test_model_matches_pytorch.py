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
from torch import nn

from tricycle import GPU_ENABLED
from tricycle.functions import Softmax
from tricycle.layers import Dense, Embedding, RMSNorm
from tricycle.loss import CrossEntropy
from tricycle.tensor import Tensor, to_tensor


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
    Tokens are a list of integers. They can be either 1d or 2d and batched
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
        tokens_,
        is_batched=len(shape) == 2,
        dtype=np.int64,
        requires_grad=False,
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
     - 1d non-batch
     - 2d non-batch
     - 2d batch
     - 3d batch
    """
    shape_ = draw(tensor_shape())
    data = draw(xp.arrays(dtype=np.float64, shape=shape_))
    match len(shape_):
        case 1:
            is_batched = False
        case 2:
            is_batched = draw(st.booleans())
        case 3:
            is_batched = True
    requires_grad = True
    if GPU_ENABLED:
        on_gpu = draw(st.booleans())
    else:
        warn("GPU_ENABLED = False so GPU tests have been disabled")
        on_gpu = False

    tensor = to_tensor(
        data,
        is_batched=is_batched,
        requires_grad=requires_grad,
    )
    if on_gpu:
        tensor = tensor.to_gpu()
    return tensor, tensor_shape


@st.composite
def embedding_shape(draw):
    """
    Embeddings are either 2d or 3d and batched
    """
    return draw(
        st.lists(
            st.integers(min_value=1, max_value=64), min_size=2, max_size=3
        )
    )


def build_tensor(shape_, is_batched):
    """
    Generate a single, initial tensor (not as the result of an operation)
    For our model, we need the following tensors:
     - 1d non-batch
     - 2d non-batch
     - 2d batch
     - 3d batch
    """
    np.random.seed(0)
    data = np.random.random(shape_).astype(np.float32)
    match len(shape_):
        case 1:
            is_batched = False
        case 2:
            is_batched = is_batched
        case 3:
            is_batched = True
    requires_grad = True

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
    if GPU_ENABLED:
        on_gpu = draw(st.booleans())
    else:
        warn("GPU_ENABLED = False so GPU tests have been disabled")
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


@given(tensor_shape(), integer(), st.booleans())
@settings(deadline=1000)
def test_tricycle_dense_matches_pytorch(in_shape, out_shape, is_batched):
    tensor = build_tensor(in_shape, is_batched)
    assume(np.isfinite(tensor.array).all())

    from_size = tensor.shape[-1]

    pt_layer = torch.nn.Linear(
        in_features=from_size, out_features=out_shape, bias=False
    )
    tr_layer = Dense(from_size=from_size, to_size=out_shape)
    tr_layer.weights = to_tensor(pt_layer.weight.detach().numpy().T)

    pt_out = pt_layer(torch.tensor(tensor.array).to(torch.float32)).to(
        torch.float16
    )
    tr_out = tr_layer(tensor)

    assert np.allclose(
        pt_out.detach().numpy(), tr_out.numpy(), rtol=1e-2, atol=1e-3
    )

    pt_out.sum().backward()
    tr_out.from_batched().sum().backward()

    assert np.allclose(
        pt_layer.weight.grad.detach().numpy().T,
        tr_layer.weights.grad.numpy(),
        rtol=1e-1,
        atol=1e-1,
    )


@given(tokens(), integer())
@settings(deadline=1000)
def test_embedding_matches(tokens_, out_shape):
    vocab_size = tokens_.array.max() + 1
    pt_layer = torch.nn.Embedding(
        num_embeddings=vocab_size, embedding_dim=out_shape
    )
    tr_layer = Embedding(from_size=vocab_size, to_size=out_shape)
    tr_layer.weights = Tensor(pt_layer.weight.detach().numpy())

    pt_out = pt_layer(torch.tensor(tokens_.array))
    tr_out = tr_layer(tokens_)

    assert np.allclose(
        pt_out.detach().numpy(), tr_out.numpy(), rtol=1e-2, atol=1e-2
    )

    pt_out.sum().backward()
    tr_out.from_batched().sum().backward()

    assert np.allclose(
        pt_layer.weight.grad.detach().numpy(),
        tr_layer.weights.grad.numpy(),
        rtol=1e-3,
        atol=1e-1,
    )


@given(tensor_shape(force_divisible_by_32=True), st.booleans())
@settings(deadline=1000)
@example(in_shape=[1, 1, 1, 128], is_batched=True)
def test_tricycle_softmax_matches_pytorch(in_shape, is_batched):
    tensor = build_tensor(in_shape, is_batched)
    assume(np.isfinite(tensor.array).all())

    tensor.requires_grad = True

    pt_layer = torch.nn.functional.softmax
    tr_layer = Softmax()

    pt_input = torch.tensor(tensor.array, requires_grad=True)

    pt_out = pt_layer(pt_input.to(torch.float32), dim=-1)
    tr_out = tr_layer(tensor)

    assert np.allclose(
        pt_out.detach().numpy(), tr_out.numpy(), rtol=1e-3, atol=1e-3
    )

    pt_out.sum().backward()
    tr_out.from_batched().sum().backward()

    assert np.allclose(
        pt_input.grad.detach().numpy(),
        tensor.grad.numpy(),
        rtol=1e-2,
        atol=1e-3,
    )


@given(tensor_shape(), st.booleans())
@example(in_shape=[2, 2, 4], is_batched=False)
def test_crossentropy_matches(in_shape, is_batched):
    y_pred = build_tensor(in_shape, is_batched)
    y_true = np.random.randint(0, in_shape[-1], size=in_shape[:-1])
    y_true = to_tensor(y_true, is_batched=is_batched, dtype=int)
    assume(np.isfinite(y_pred.array).all())

    tr_out = CrossEntropy()(y_true, y_pred).from_batched()
    if len(in_shape) > 1:
        tr_out = tr_out.mean()

    if len(in_shape) == 1:
        p_y_pred = copy(y_pred.array)
    if len(in_shape) == 2:
        p_y_pred = copy(y_pred.array)
    if len(in_shape) == 3:
        p_y_pred = copy(y_pred.array).transpose(0, -1, 1)
    p_y_pred = torch.tensor(p_y_pred, requires_grad=True)
    p_y_true = torch.tensor(y_true.array, dtype=torch.long)

    p_out = torch.nn.CrossEntropyLoss()(
        input=p_y_pred,
        target=p_y_true,
    )

    assert tr_out.close_to(p_out.detach().numpy().item(), rtol=1e-3)

    tr_out.backward()
    p_out.backward()

    p_grad = p_y_pred.grad.detach().numpy()
    if len(in_shape) == 3:
        p_grad = p_grad.transpose(0, -1, 1)
    assert y_pred.grad.close_to(p_grad, rtol=1e-2)


# @given(tensor_shape(), st.booleans())
# @example(in_shape=[2, 2, 4], is_batched=False)
# def test_rotary_encodings_match(in_shape, is_batched):
#     y_pred = build_tensor(in_shape, is_batched)
#     y_true = np.random.randint(0, in_shape[-1], size=in_shape[:-1])
#     y_true = to_tensor(y_true, is_batched=is_batched, dtype=int)
#     assume(np.isfinite(y_pred.array).all())
#
# tr_out = CrossEntropy()(y_true, y_pred).from_batched()
# if len(in_shape) > 1:
#     tr_out = tr_out.mean()
#
# if len(in_shape) == 1:
#     p_y_pred = copy(y_pred.array)
# if len(in_shape) == 2:
#     p_y_pred = copy(y_pred.array)
# if len(in_shape) == 3:
#     p_y_pred = copy(y_pred.array).transpose(0, -1, 1)
# p_y_pred = torch.tensor(p_y_pred, requires_grad=True)
# p_y_true = torch.tensor(y_true.array, dtype=torch.long)
#
# p_out = torch.nn.CrossEntropyLoss()(
#     input=p_y_pred,
#     target=p_y_true,
# )
#
# assert tr_out.close_to(p_out.detach().numpy().item())
#
# tr_out.backward()
# p_out.backward()
#
# p_grad = p_y_pred.grad.detach().numpy()
# if len(in_shape) == 3:
#     p_grad = p_grad.transpose(0, -1, 1)
# assert y_pred.grad.close_to(p_grad)


# reference implementation of rmsnorm: https://github.com/meta-llama/llama/blob/be327c427cc5e89cc1d3ab3d3fec4484df771245/llama/model.py#L34
class PytorchRMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


@given(tensor_shape(), st.booleans())
@example(in_shape=[32, 1], is_batched=False)
@settings(deadline=10_000)
def test_rms_norm_matches(in_shape, is_batched):
    tr_tensor = build_tensor(in_shape, is_batched)
    assume(np.isfinite(tr_tensor.array).all())
    pt_tensor = torch.tensor(copy(tr_tensor.array), requires_grad=True)

    embedding_dim = tr_tensor.shape[-1]
    tr_layer = RMSNorm(embedding_dim)
    tr_out = tr_layer(tr_tensor).from_batched().mean()

    pt_layer = PytorchRMSNorm(embedding_dim)
    pt_out = pt_layer(pt_tensor).mean()

    assert tr_out.close_to(pt_out.detach().numpy(), rtol=1e-2, atol=1e-3)

    tr_out.backward()
    pt_out.backward()

    pt_weight_grad = pt_layer.weight.grad.detach().numpy()
    tr_weight_grad = tr_layer.weights.grad
    assert tr_weight_grad.close_to(pt_weight_grad, rtol=1e-2, atol=1e-3)

    tr_grad = tr_tensor.grad
    pt_grad = pt_tensor.grad.detach().numpy()
    assert tr_grad.close_to(pt_grad, atol=1e0, rtol=1e-1)
