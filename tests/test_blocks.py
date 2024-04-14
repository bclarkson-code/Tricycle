from copy import copy

import numpy as np
import torch

from tricycle.blocks import (
    GPT2TransformerBlock,
    MLPBlock,
    MultiHeadSelfAttention,
    build_mask,
    masked_fill,
)
from tricycle.einsum import Einsum
from tricycle.functions import softmax
from tricycle.tensor import to_tensor


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

    x = torch.from_numpy(in_tensor._data)

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


def pytorch_attention(q, k, v, B, T, C, n_head, block_size=32):
    k = k.view(B, T, n_head, C // n_head).transpose(1, 2)  # (B, nh, T, hs)
    q = q.view(B, T, n_head, C // n_head).transpose(1, 2)  # (B, nh, T, hs)
    v = v.view(B, T, n_head, C // n_head).transpose(1, 2)  # (B, nh, T, hs)
    y = torch.nn.functional.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=None,
        dropout_p=0,
        is_causal=True,
    )
    y = y.transpose(1, 2).contiguous().view(B, T, C)
    return y


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

    np.random.seed(0)
    # random input tensor
    in_tensor = np.random.uniform(
        -5, 5, (batch_size, n_tokens, projected_size)
    )
    in_tensor = to_tensor(in_tensor).to_vector()

    x = torch.from_numpy(in_tensor._data)

    qu, k, v = x.split(embedding_dim, dim=-1)  # pytorch
    query, key, value = in_tensor.split(3, axis=1)  # tricycle

    assert query.close_to(qu)
    assert key.close_to(k)
    assert value.close_to(v)

    andrej_result = andrej_attention(
        q=copy(qu),
        k=copy(k),
        v=copy(v),
        B=B,
        T=T,
        C=C,
        n_head=n_heads,
        block_size=context_window,
    ).numpy()

    pytorch_result = pytorch_attention(
        q=copy(qu),
        k=copy(k),
        v=copy(v),
        B=B,
        T=T,
        C=C,
        n_head=n_heads,
        block_size=context_window,
    )

    tricycle_attention = MultiHeadSelfAttention(
        embedding_dim=embedding_dim,
        n_heads=n_heads,
        context_window=context_window,
        residual_dropout_prob=0,
        attention_dropout_prob=0,
    )
    tricycle_result = tricycle_attention._attention(
        query=query, key=key, value=value
    ).from_vector()

    assert np.allclose(andrej_result, pytorch_result, rtol=1e-3)
    assert tricycle_result.close_to(andrej_result)
    assert tricycle_result.close_to(pytorch_result)


def andrej_attention_block(
    x, B, T, C, n_head, c_attn, c_proj, n_embd, block_size=32
):
    """
    Andrej Karpathy's implementation of an attention block from nanogpt
    """
    q, k, v = c_attn(x).split(n_embd, dim=2)
    y = andrej_attention(q, k, v, B, T, C, n_head, block_size)
    return c_proj(y)


def test_attention_block():
    """
    Compare Tricycle attention with pytorch's MultiheadAttention
    """
    n_heads = 3
    embedding_dim = 15
    n_tokens = 7
    batch_size = 11
    context_window = 32

    np.random.seed(0)

    x = np.random.normal(size=(batch_size, n_tokens, embedding_dim))

    in_projection_weights = np.random.normal(
        0, 1, (embedding_dim, embedding_dim * 3)
    )
    out_projection_weights = np.random.normal(
        0, 1, (embedding_dim, embedding_dim)
    )

    tricycle_attention = MultiHeadSelfAttention(
        embedding_dim=embedding_dim,
        n_heads=n_heads,
        context_window=context_window,
        residual_dropout_prob=0,
        attention_dropout_prob=0,
    )
    tricycle_attention.in_projection.weights = to_tensor(
        in_projection_weights, name="in_proj"
    )
    tricycle_attention.out_projection.weights = to_tensor(
        out_projection_weights, name="out_proj"
    )

    in_tensor = to_tensor(x, requires_grad=False).to_vector()
    tricycle_result = tricycle_attention(in_tensor)

    c_attn = torch.nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
    c_attn.weight = torch.nn.Parameter(torch.tensor(in_projection_weights.T))
    c_proj = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)
    c_proj.weight = torch.nn.Parameter(torch.tensor(out_projection_weights.T))

    andrej_result = andrej_attention_block(
        torch.tensor(x),
        batch_size,
        n_tokens,
        embedding_dim,
        n_heads,
        c_attn,
        c_proj,
        embedding_dim,
        block_size=32,
    )

    assert tricycle_result.close_to(andrej_result.detach().numpy())

    tricycle_loss = tricycle_result.from_vector().e("abc->")
    andrej_loss = andrej_result.sum()

    assert tricycle_loss.close_to(andrej_loss.detach().numpy())

    tricycle_loss.backward()
    andrej_loss.backward()

    assert not tricycle_attention.out_projection.weights.is_vector
    tricycle_out_weights = tricycle_attention.out_projection.weights.grad
    tricycle_out_weights = tricycle_out_weights.from_vector().e("abc->bc")

    assert tricycle_out_weights.close_to(c_proj.weight.grad.T.numpy())

    tricycle_in_weights = tricycle_attention.in_projection.weights.grad
    tricycle_in_weights = tricycle_in_weights.from_vector().e("abc->bc")

    assert tricycle_in_weights.close_to(
        c_attn.weight.grad.T.numpy(), rtol=1e-3
    )


def test_MLPBlock():
    np.random.seed(0)
    in_tensor = to_tensor(np.arange(12, dtype=float).reshape(3, 4))
    block = MLPBlock(embedding_dim=4, expansion_ratio=4, dropout_prob=0.5)

    assert block.linear_1.weights.shape == (4, 16)
    assert block.linear_2.weights.shape == (16, 4)

    block.linear_1.weights = to_tensor(np.ones(block.linear_1.weights.shape))
    block.linear_2.weights = to_tensor(np.ones(block.linear_2.weights.shape))

    out_tensor = block(in_tensor.to_vector())

    assert out_tensor.shape == (3, 4)

    correct_output = np.array(
        [[96, 0, 96, 0], [0, 0, 352, 352], [608, 608, 608, 0]]
    )
    correct_output = to_tensor(correct_output)

    assert out_tensor.is_vector
    assert out_tensor.close_to(correct_output)

    out_tensor.backward()

    assert in_tensor.grad is not None
    correct_grad = to_tensor(
        [
            [-44.59691787, -44.59691787, -44.59691787, -44.59691787],
            [-248.8553654, -248.8553654, -248.8553654, -248.8553654],
            [-679.67071945, -679.67071945, -679.67071945, -679.67071945],
        ]
    )

    assert in_tensor.grad.close_to(correct_grad)


def test_GPT2TransformerBlock():
    np.random.seed(0)
    batch_size = 11
    n_tokens = 5
    n_heads = 3
    embedding_dim = 7 * n_heads

    in_tensor = to_tensor(
        np.random.random((batch_size, n_tokens, embedding_dim)), is_vector=True
    )
    block = GPT2TransformerBlock(
        embedding_dim=embedding_dim,
        n_heads=3,
        expansion_ratio=4,
        context_window=32,
    )

    out_tensor = block(in_tensor.to_vector())

    assert out_tensor.shape == (batch_size, n_tokens, embedding_dim)

    out_tensor.backward()

    assert in_tensor.grad is not None
    assert in_tensor.grad.shape == in_tensor.shape
