import numpy as np
import pytest
import torch

from tricycle.binary import badd
from tricycle.einsum import Einsum, Subscript
from tricycle.layers import (
    Dense,
    MultiHeadSelfAttention,
    Sequential,
    build_mask,
    masked_fill,
)
from tricycle.ops import reshape, softmax, split, to_tensor


def test_dense_layer():
    layer = Dense(10, 8)

    assert layer.weights.shape == (10, 8)

    x_in = to_tensor(np.ones(10))

    x_out = layer(x_in)
    assert x_out.shape == (8,)


def test_sequential_layer():
    layer1 = Dense(10, 8)
    layer2 = Dense(8, 4)

    model = Sequential(layer1, layer2)

    assert model.layers[0].weights.shape == (10, 8)
    assert model.layers[1].weights.shape == (8, 4)

    x_in = to_tensor(np.ones(10))

    x_out = model(x_in)
    assert x_out.shape == (4,)


def test_attention_individually():
    """
    This operation is pretty complex so we'll perform each stage
    with pytorch and then compare the results. Here, I'm comparing
    with Andrej Karpathy's implementation from NanoGPT
    For this test, we're doing everything non-vectorised
    """
    # setup
    embedding_dim = 15
    n_heads = 3
    n_tokens = 7
    projected_size = embedding_dim * 3
    context_window = 32
    head_size = embedding_dim // n_heads
    head_shape = (n_tokens, n_heads, head_size)
    out_shape = (n_tokens, embedding_dim)
    T = n_tokens
    C = embedding_dim

    # random input tensor
    in_tensor = np.random.uniform(-5, 5, (n_tokens, projected_size))
    in_tensor = to_tensor(in_tensor)

    x = torch.from_numpy(in_tensor)

    qu, k, v = x.split(embedding_dim, dim=1)  # pytorch
    query, key, value = in_tensor.split(3, axis=1)  # tricycle

    assert query.close_to(qu)
    assert key.close_to(k)
    assert value.close_to(v)

    # pytorch
    k = k.view(T, n_heads, C // n_heads)
    qu = qu.view(T, n_heads, C // n_heads)
    v = v.view(T, n_heads, C // n_heads)
    k = k.transpose(-3, -2)
    qu = qu.transpose(-3, -2)
    v = v.transpose(-3, -2)

    # tricycle
    key = key.reshape(head_shape).e("TNH -> NTH")
    query = query.reshape(head_shape).e("TNH -> NTH")
    value = value.reshape(head_shape).e("TNH -> NTH")

    assert query.close_to(qu)
    assert key.close_to(k)
    assert value.close_to(v)

    # pytorch
    att = qu @ k.transpose(-2, -1)
    att *= 1 / np.sqrt(k.size(-1))

    # tricycle
    attention = Einsum("NIh, NJh -> NIJ")(query, key) / np.sqrt(head_size)

    assert attention.close_to(att)

    # pytorch
    bias = torch.tril(torch.ones(context_window, context_window)).view(
        1, context_window, context_window
    )
    att = att.masked_fill(bias[:, :T, :T] == 0, float("-inf"))

    # tricycle
    mask = build_mask(context_window)
    attention = masked_fill(attention, (n_tokens, n_tokens), mask)

    assert attention.close_to(att)

    # pytorch
    att = torch.softmax(att, dim=-1)

    # tricycle
    attention = softmax(attention)

    assert attention.close_to(att.numpy())

    # pytorch
    att = att @ v
    att = att.transpose(0, 1).contiguous()

    # tricycle
    attention = Einsum("NIj, NjH -> INH")(attention, value)

    assert attention.close_to(att.numpy())

    # pytorch
    att = att.view(T, C)

    # tricycle
    attention = attention.reshape(out_shape)

    assert attention.close_to(att.numpy())


def andrej_attention(q, k, v, B, T, C, n_head, block_size=32):
    """
    Andrej Karpathy's implementation of attention from nanogpt
    """
    import math

    from torch.nn import functional as F

    bias = torch.tril(torch.ones(block_size, block_size)).view(
        1, 1, block_size, block_size
    )
    k = k.view(B, T, n_head, C // n_head).transpose(1, 2)  # (B, nh, T, hs)
    q = q.view(B, T, n_head, C // n_head).transpose(1, 2)  # (B, nh, T, hs)
    v = v.view(B, T, n_head, C // n_head).transpose(1, 2)  # (B, nh, T, hs)

    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    att = att.masked_fill(bias[:, :, :T, :T] == 0, float("-inf"))
    att = F.softmax(att, dim=-1)
    y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
    return y.transpose(1, 2).contiguous().view(B, T, C)


def test_attention_combined():
    """
    Compare Tricycle's attention with Andrej's
    """
    n_heads = 3
    embedding_dim = 15
    n_tokens = 7
    batch_size = 11
    projected_size = embedding_dim * 3
    context_window = 32
    B = batch_size
    T = n_tokens
    C = embedding_dim

    # random input tensor
    in_tensor = np.random.uniform(
        -5, 5, (batch_size, n_tokens, projected_size)
    )
    in_tensor = to_tensor(in_tensor).to_vector()

    x = torch.from_numpy(in_tensor)

    qu, k, v = x.split(embedding_dim, dim=-1)  # pytorch
    query, key, value = in_tensor.split(3, axis=1)  # tricycle

    assert query.close_to(qu)
    assert key.close_to(k)
    assert value.close_to(v)

    andrej_result = andrej_attention(
        qu, k, v, B, T, C, n_heads, context_window
    )

    tricycle_attention = MultiHeadSelfAttention(
        embedding_dim=embedding_dim,
        n_heads=n_heads,
        context_window=context_window,
        dropout=0,
    )
    tricycle_result = tricycle_attention._attention(
        query, key, value
    ).from_vector()

    assert tricycle_result.close_to(andrej_result, rtol=1e0)


@pytest.mark.skip
def test_attention_block():
    """
    Compare Tricycle attention with pytorch's MultiheadAttention
    """
    embedding_dim = 10
    n_heads = 2
    n_tokens = 10

    np.random.seed(1)

    key = np.random.uniform(-5, 5, (n_tokens, embedding_dim))
    query = np.random.uniform(-5, 5, (n_tokens, embedding_dim))
    value = np.random.uniform(-5, 5, (n_tokens, embedding_dim))

    k = torch.from_numpy(key)
    qu = torch.from_numpy(query)
    v = torch.from_numpy(value)

    pytorch_attention = torch.nn.functional.scaled_dot_product_attention(
        qu,
        k,
        v,
        attn_mask=None,
        dropout_p=0,
        is_causal=True,
    )
    attention = MultiHeadSelfAttention(
        embedding_dim, n_heads, context_window=32, dropout=0
    )
    tricycle_attention = attention._attention(
        to_tensor(k), to_tensor(qu), to_tensor(v)
    )
    andrej = andrej_attention(qu, k, v, n_tokens, attention.mask)

    breakpoint()
    assert np.allclose(tricycle_attention, pytorch_attention.numpy())
