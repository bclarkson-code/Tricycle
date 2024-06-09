import numpy as np
import torch

from tricycle.attention import Attention
from tricycle.tensor import to_tensor


def pytorch_attention(q, k, v, B, T, C, n_head):
    k = k.view(B, T, n_head, C // n_head).transpose(1, 2)  # (B, nh, T, hs)
    q = q.view(B, T, n_head, C // n_head).transpose(1, 2)  # (B, nh, T, hs)
    v = v.view(B, T, n_head, C // n_head).transpose(1, 2)  # (B, nh, T, hs)
    # return k
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


def andrej_attention(q, k, v, B, T, C, n_head, block_size=32, bias=None):
    """
    Andrej Karpathy's implementation of attention from nanogpt
    """
    import math

    from torch.nn import functional as F

    if bias is None:
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
    context_window = n_tokens
    B = batch_size
    T = n_tokens
    C = embedding_dim

    np.random.seed(0)
    # random input tensor
    in_tensor = np.random.uniform(
        -5, 5, (batch_size, n_tokens, projected_size)
    )
    in_tensor = to_tensor(in_tensor).to_vector()

    x = torch.from_numpy(in_tensor.array)
    x.requires_grad = True

    qu, k, v = x.split(embedding_dim, dim=-1)  # pytorch

    pytorch_result = andrej_attention(
        q=qu,
        k=k,
        v=v,
        B=B,
        T=T,
        C=C,
        n_head=n_heads,
    )

    tricycle_attention = Attention(
        embedding_dim=embedding_dim,
        n_heads=n_heads,
        context_window=context_window,
    )
    tricycle_result = tricycle_attention(in_tensor).from_vector()

    assert tricycle_result.close_to(
        pytorch_result.detach().numpy(), equal_nan=True, rtol=1e-3, atol=1e-5
    )

    tricycle_result.from_vector().sum().backward()
    pytorch_result.sum().backward()

    assert in_tensor.grad.close_to(
        x.grad.detach().numpy(), atol=1e-5, equal_nan=True, rtol=1e-4
    )
