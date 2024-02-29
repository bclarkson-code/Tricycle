import numpy as np
import torch

from tricycle.binary import badd
from tricycle.layers import Dense, MultiHeadSelfAttention, Sequential
from tricycle.ops import einsum, reshape, softmax, split, to_tensor


def test_dense_layer():
    layer = Dense(10, 8)

    assert layer.weights.shape == (10, 8)

    x_in = np.ones(10)

    x_out = layer(x_in)
    assert x_out.shape == (8,)


def test_sequential_layer():
    layer1 = Dense(10, 8)
    layer2 = Dense(8, 4)

    seq = Sequential(layer1, layer2)

    assert seq.layers[0].weights.shape == (10, 8)
    assert seq.layers[1].weights.shape == (8, 4)

    x_in = np.ones(10)

    x_out = seq(x_in)
    assert x_out.shape == (4,)


def test_attention_individually():
    """
    This operation is pretty complex so we'll perform each stage
    with pytorch and then compare the results
    """
    # setup
    embedding_dim = 10
    n_heads = 2
    n_tokens = 10
    context_window = 32
    projected_size = embedding_dim * 3

    # random input tensor
    in_tensor = np.random.uniform(-5, 5, (n_tokens, projected_size))

    T = n_tokens
    C = embedding_dim

    x = torch.from_numpy(in_tensor)

    qu, k, v = x.split(embedding_dim, dim=1)  # pytorch
    query, key, value = split(in_tensor, n_splits=3)  # tricycle
    assert np.allclose(query, qu.numpy()), (query, qu.numpy())
    assert np.allclose(key, k.numpy())
    assert np.allclose(value, v.numpy())

    # pytorch
    k = k.view(T, n_heads, C // n_heads)
    qu = qu.view(T, n_heads, C // n_heads)
    v = v.view(T, n_heads, C // n_heads)

    # tricycle
    head_size = embedding_dim // n_heads
    key = reshape(key, (n_tokens, n_heads, head_size))
    query = reshape(query, (n_tokens, n_heads, head_size))
    value = reshape(value, (n_tokens, n_heads, head_size))

    assert np.allclose(key, k.numpy())
    assert np.allclose(query, qu.numpy())
    assert np.allclose(value, v.numpy())

    # pytorch
    k = k.transpose(-3, -2)
    qu = qu.transpose(-3, -2)
    v = v.transpose(-3, -2)

    # tricycle
    swap = einsum("tnh->nth")
    key = swap(key)
    query = swap(query)
    value = swap(value)

    assert np.allclose(key, k.numpy())
    assert np.allclose(query, qu.numpy())
    assert np.allclose(value, v.numpy())

    # pytorch
    att = qu @ k.transpose(-2, -1)

    # tricycle
    attend = einsum("nih,njh->nij")
    attention = attend(query, key)

    assert np.allclose(attention, att.numpy())

    # pytorch
    att *= 1 / np.sqrt(k.size(-1))

    # tricycle
    attention = attention / np.sqrt(head_size)

    assert np.allclose(attention, att.numpy())

    # pytorch
    bias = torch.tril(torch.ones(context_window, context_window)).view(
        1, context_window, context_window
    )
    att = att.masked_fill(bias[:, :T, :T] == 0, float("-inf"))

    # tricycle
    mask = np.ones((context_window, context_window))
    idx = np.tril(mask.astype(bool))
    mask[~idx] = -np.inf
    mask[idx] = 0
    mask = np.stack([mask[:T, :T]] * attention.shape[0])
    mask = to_tensor(mask, requires_grad=False, name="mask")

    # TODO: check if this breaks the gradients
    attention = badd(attention, mask)

    assert np.allclose(att.numpy(), attention)

    # pytorch
    att = torch.softmax(att, dim=-1)

    # tricycle
    attention = softmax(attention)

    assert np.allclose(att.numpy(), attention)

    # pytorch
    att = att @ v

    # tricycle
    attention = einsum("nij,njh->nih")(attention, value)

    assert np.allclose(att.numpy(), attention)

    # pytorch
    att = att.transpose(0, 1).contiguous().view(T, C)

    # tricycle
    attention = swap(attention)
    attention = reshape(attention, (n_tokens, embedding_dim))

    assert np.allclose(att.numpy(), attention)


def andrej_attention(q, k, v, T, bias):
    import math

    from torch.nn import functional as F

    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    att = att.masked_fill(bias[:, :, :T, :T] == 0, float("-inf"))
    att = F.softmax(att, dim=-1)
    return att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)


def test_attention_combined():
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
    tricycle_attention = attention._attention(to_tensor(k), to_tensor(qu), to_tensor(v))
    andrej = andrej_attention(qu, k, v, n_tokens, attention.mask)

    breakpoint()
    assert np.allclose(tricycle_attention, pytorch_attention.numpy())
