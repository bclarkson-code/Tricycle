import numpy as np

from tricycle.attention import Attention, TritonAttention
from tricycle.tensor import Tensor
from tricycle.utils import UseMixedPrecision

N_LOOPS = 100
BATCH_SIZE = 16
N_HEADS = 12
N_TOKENS = 1024
EMBEDDING_DIM = 768


vanilla_layer = Attention(
    embedding_dim=EMBEDDING_DIM,
    n_heads=N_HEADS,
    context_window=N_TOKENS,
).to_gpu()

triton_layer = TritonAttention(
    batch_size=BATCH_SIZE,
    embedding_dim=EMBEDDING_DIM,
    n_heads=N_HEADS,
    n_tokens=N_TOKENS,
).to_gpu()


def build_tensor(seed=20):
    np.random.seed(seed)
    shape = BATCH_SIZE, N_TOKENS, EMBEDDING_DIM * 3
    return Tensor(
        (np.random.random(shape).astype(np.float16) * 2 - 1),
        is_batched=True,
        dtype=np.float16,
    ).to_gpu()


tensor = build_tensor()

with UseMixedPrecision():
    output = vanilla_layer(tensor)
    output.backward()

    output = triton_layer(tensor)
    output.backward()


def run_loops(tensor, layer):
    with UseMixedPrecision():
        for _ in range(N_LOOPS):
            output = layer(tensor)
            output.backward()


def bench_vanilla_attention():
    run_loops(tensor, vanilla_layer)


def bench_triton_attention():
    run_loops(tensor, triton_layer)


__benchmarks__ = [
    (
        bench_vanilla_attention,
        bench_triton_attention,
        "vanilla vs triton attention",
    ),
]
