import itertools
import math
from copy import deepcopy
from functools import partial

import numpy as np
import pandas as pd
import pytest
import torch
import triton
from matplotlib import pyplot as plt

from tricycle.activation import (
    CudaGeLU,
    CudaReLU,
    GeLU,
    ReLU,
    TritonGeLU,
    TritonRelu,
)
from tricycle.attention import Attention, TritonAttention
from tricycle.context import TRICYCLE_CONTEXT
from tricycle.kernels import (  # single_batched_matmul_kernel_1,; single_batched_matmul_kernel_2,
    TritonAttentionRef,
)
from tricycle.layers import Dense, TritonDense
from tricycle.tensor import Tensor
from tricycle.utils import UseMixedPrecision


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["n_tokens"],
        # Modify x_vals to account for 3D shape
        x_vals=[
            2**i for i in range(8, 13)
        ],  # Reduced range since total elements will be cubed
        x_log=True,
        line_arg="provider",
        line_vals=["triton", "tricycle"],
        line_names=["Triton", "Tricycle"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="GB/s",
        plot_name="attention-performance",
        args={},
    )
)
def benchmark(n_tokens, provider):
    # Create 3D input matrices

    quantiles = [0.5, 0.2, 0.8]
    batch_size, n_heads, n_tokens, head_size = (
        4,
        12,
        n_tokens,
        64,
    )

    causal = True
    tensor = np.random.normal(
        loc=0, scale=0.5, size=(batch_size, n_tokens, n_heads * head_size * 3)
    ).astype(np.float16)
    tensor = Tensor(tensor, dtype=np.float16, is_batched=True)
    tensor = tensor.to_gpu()

    match provider:
        case "triton":

            layer = TritonAttention(
                batch_size=batch_size,
                embedding_dim=n_heads * head_size,
                n_heads=n_heads,
                n_tokens=n_tokens,
            )

        case "tricycle":

            layer = Attention(
                embedding_dim=n_heads * head_size,
                n_heads=n_heads,
                context_window=n_tokens,
            )

        case _:
            raise ValueError(provider)
    layer.to_gpu()

    def fn():
        with UseMixedPrecision():
            layer(tensor)

        # layer.backward(y)

    ms, min_ms, max_ms = triton.testing.do_bench(
        lambda: fn(), quantiles=quantiles
    )

    # Update gbps calculation to account for 3D size
    def gbps(ms):
        flops_per_matmul = (
            2.0 * batch_size * n_heads * n_tokens * n_tokens * head_size
        )
        total_flops = 2 * flops_per_matmul
        if causal:
            total_flops *= 0.5
        return total_flops * 1e-9 / (ms * 1e-3)

    return gbps(ms), gbps(max_ms), gbps(min_ms)


def andrej_attention(
    tensor, B, T, C, n_head, block_size=32, sm_scale=0.5, bias=None
):
    """
    Andrej Karpathy's implementation of attention from nanogpt
    """
    import math

    from torch.nn import functional as F

    if bias is None:
        bias = (
            torch.tril(torch.ones(block_size, block_size))
            .view(1, 1, block_size, block_size)
            .to(tensor.device)
        )

    q, k, v = tensor.split(C, dim=-1)
    k = k.view(B, T, n_head, C // n_head).transpose(1, 2)  # (B, nh, T, hs)
    q = q.view(B, T, n_head, C // n_head).transpose(1, 2)  # (B, nh, T, hs)
    v = v.view(B, T, n_head, C // n_head).transpose(1, 2)  # (B, nh, T, hs)

    att = q @ k.transpose(-2, -1)
    att *= sm_scale
    mask = torch.tril(torch.ones((T, T)).cuda())
    att[:, :, mask == 0] = float("-inf")
    att = F.softmax(att.to(torch.float32), dim=-1).half()
    y = att @ v
    return y.transpose(1, 2).contiguous().view(B, T, C)


def compare_outputs(n_tokens):
    """
    Compares the outputs of Triton and Tricycle attention implementations
    to ensure they produce the same results within specified tolerance.

    Args:
        n_tokens: Number of tokens for the test
        atol: Absolute tolerance for comparison
        rtol: Relative tolerance for comparison

    Returns:
        bool: True if outputs match within tolerance, False otherwise
        dict: Dictionary with error statistics
    """
    # Setup parameters (same as in benchmark function)
    batch_size, n_heads, head_size = 4, 12, 64

    DEVICE = torch.device("cuda:0")
    dtype = torch.float16

    torch.manual_seed(20)

    # Create input tensor
    tensor = (
        torch.empty(
            (batch_size, n_tokens, n_heads * head_size * 3),
            dtype=dtype,
            device=DEVICE,
        )
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    tricycle_tensor = Tensor(
        tensor.clone().detach().cpu(), is_batched=True, dtype=np.float16
    ).to_gpu()
    sm_scale = 1 / math.sqrt(head_size)

    ref_out = andrej_attention(
        tensor,
        B=batch_size,
        T=n_tokens,
        C=head_size * n_heads,
        n_head=n_heads,
        block_size=n_tokens,
        sm_scale=sm_scale,
    )
    grad = torch.rand_like(ref_out)

    ref_out.backward(grad)
    ref_tensor_grad, tensor.grad = tensor.grad.clone(), None

    # get triton output
    triton_ref = TritonAttentionRef()
    triton_output = triton_ref.forward(
        tensor=tensor.clone(),
        causal=True,
        sm_scale=sm_scale,
        batch_size=batch_size,
        n_heads=n_heads,
        head_size=head_size,
        n_tokens=n_tokens,
    ).half()
    tri_tensor_grad = triton_ref.backward(grad.contiguous())

    # compare outputs
    assert torch.allclose(triton_output, ref_out, rtol=0, atol=1e-2)

    grad_matches = torch.allclose(
        tri_tensor_grad, ref_tensor_grad, rtol=0, atol=1e-2
    )
    assert grad_matches

    # Get Tricycle output
    tricycle_layer = TritonAttention(
        batch_size=batch_size,
        n_tokens=n_tokens,
        embedding_dim=n_heads * head_size,
        n_heads=n_heads,
    )
    tricycle_layer.sm_scale = sm_scale
    tricycle_layer.to_gpu()
    TRICYCLE_CONTEXT.use_mixed_precision = True
    tricycle_output = tricycle_layer.forward(tricycle_tensor)
    tricycle_output.backward(Tensor(grad.clone().cpu().numpy()).to_gpu())
    tricycle_tensor_grad = tricycle_tensor.grad

    assert tricycle_output.close_to(triton_output, rtol=0, atol=1e-2)

    # compare with triton reference
    grad_matches = tricycle_tensor_grad.close_to(
        tri_tensor_grad, rtol=0, atol=1e-2
    )
    assert grad_matches

    return True


def test_all_sizes():
    """
    Tests output comparison for all sizes used in the benchmark.

    Args:
        atol: Absolute tolerance for comparison
        rtol: Relative tolerance for comparison

    Returns:
        dict: Results for each tested size
    """
    # Same token sizes as in the benchmark

    results = {}
    for n_tokens in [256, 512, 1024]:

        is_match = compare_outputs(n_tokens)
        results[n_tokens] = {"match": is_match}

        if not is_match:
            print(f"❌ Outputs do not match for n_tokens = {n_tokens}")
        else:
            print(
                f"✅ Outputs match within tolerance for n_tokens = {n_tokens}"
            )

    return results


if __name__ == "__main__":
    # Test with default tolerances
    results = test_all_sizes()

    # # Print summary
    # all_match = all(result["match"] for result in results.values())

    # if all_match:
    #     print("\n✓ All tests passed! Outputs match within tolerance.")
    # else:
    #     print("\n❌ Some tests failed. Check individual results.")

    benchmark.run(print_data=True, show_plots=True)
    plt.savefig("fig.png")
