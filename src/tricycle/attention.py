from math import sqrt

import cupy as cp
import numpy as np

from tricycle.einsum import Einsum
from tricycle.functions import Softmax
from tricycle.initialisers import init_xavier
from tricycle.layers import Dense, Dropout, Layer, LayerNorm  # noqa E501
from tricycle.ops import Op
from tricycle.optimisers import Optimiser
from tricycle.tensor import Tensor, select_backend, to_tensor

EMBEDDING_DIM = 384
N_HEADS = 6
CONTEXT_WINDOW = 256
DROPOUT_PROB = 0


def build_mask(context_window: int, n_heads: int) -> Tensor:
    """
    Build an attention mask to stop the model from being able to see
    future tokens
    """
    mask = np.ones((context_window, context_window), dtype=bool)
    idx = np.tril(mask)
    mask = np.stack([~idx] * n_heads)
    return mask


def masked_fill(
    tensor: Tensor, mask_shape: tuple[int, int], full_mask: Tensor
):
    """
    Apply an attention_mask to a tensor
    """
    xp = tensor.xp
    repeats = tensor.shape[1] if tensor.is_vector else tensor.shape[0]
    mask = xp.stack(
        [full_mask[: mask_shape[0], : mask_shape[1]]._data] * repeats
    )
    mask = to_tensor(mask, requires_grad=False, name="mask")
    result = tensor + mask
    result.name = "masked"
    return result


attention_dropout = Dropout(DROPOUT_PROB)


def attention_v1(key, query, value, mask):
    # xp = select_backend(key._data, query._data, value._data)
    # reshape into n_heads x embedding_dim
    head_size = EMBEDDING_DIM // N_HEADS
    n_tokens = key.shape[1] if key.is_vector else key.shape[0]
    head_shape = (
        n_tokens,  # number of tokens
        N_HEADS,  # number of heads
        head_size,  # embedding per head
    )
    out_shape = (n_tokens, EMBEDDING_DIM)

    # reshape and reorder the heads
    key = key.reshape(head_shape).e("TNH -> NTH")
    query = query.reshape(head_shape).e("TNH -> NTH")
    value = value.reshape(head_shape).e("TNH -> NTH")

    # attend
    divisor = sqrt(head_size)
    attention = Einsum("NIh, NJh -> NIJ")(query, key)
    attention = attention / divisor

    # mask and softmax
    attention = masked_fill(attention, (n_tokens, n_tokens), mask)

    attention = Softmax()(attention)

    attention = attention_dropout(attention)

    # smush the heads back together
    out_shape = (n_tokens, EMBEDDING_DIM)
    return Einsum("NIj, NjH -> INH")(attention, value).reshape(out_shape)


class Attention(Op):
    def __init__(
        self,
        embedding_dim: int,
        n_heads: int,
        dropout_prob: float,
        context_window: int,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads
        self.context_window = context_window
        self.dropout_prob = dropout_prob
        self.dropout = Dropout(dropout_prob)
        self.mask = build_mask(
            context_window=self.context_window, n_heads=self.n_heads
        )
        self._grad = None

    def backward(self, grad: Tensor):
        xp = grad.xp
        in_shape = (self.batch_size, self.context_window, self.embedding_dim)

        attention = grad._data

        # smush TODO: come up with a better name
        attention = attention.reshape(
            (
                self.batch_size,
                self.context_window,
                self.n_heads,
                self.head_size,
            )
        )
        value = xp.einsum("BNIj, BINH -> BNjH", self._before_smush, attention)

        attention = xp.einsum("BINH, BNjH -> BNIj", attention, self._value)

        # dropout
        if self.dropout_prob:
            attention = xp.where(self._dropout_mask, 0, attention)

        # softmax
        inner = xp.sum(attention * self._before_smush, axis=-1, keepdims=True)
        attention = self._before_smush * (attention - inner)

        # mask
        attention = xp.where(
            self.mask[:, : self.n_tokens, : self.n_tokens], 0, attention
        )
        attention /= self.divisor

        # attend
        query = xp.einsum("BNIJ, BNJh -> BNIh", attention, self._key)
        key = xp.einsum("BNIh, BNIJ -> BNJh", self._query, attention)

        # reshape + reorder
        key = xp.einsum("BNTH->BTNH", key)
        query = xp.einsum("BNTH->BTNH", query)
        value = xp.einsum("BNTH->BTNH", value)

        key = key.reshape(in_shape)
        query = query.reshape(in_shape)
        value = value.reshape(in_shape)

        if self._grad is None:
            self._grad = xp.zeros(
                (self.batch_size, self.context_window, self.embedding_dim * 3)
            )
        self._grad[:, :, : self.embedding_dim] = query
        self._grad[:, :, self.embedding_dim : self.embedding_dim * 2] = key
        self._grad[:, :, self.embedding_dim * 2 :] = value

        return to_tensor(self._grad)

    def forward(self, tensor: Tensor):
        xp = tensor.xp

        assert tensor.is_vector

        self._input = tensor
        query = tensor[:, :, : self.embedding_dim]
        key = tensor[:, :, self.embedding_dim : self.embedding_dim * 2]
        value = tensor[:, :, self.embedding_dim * 2 :]

        # Figure out how big everything is
        self.batch_size = key._data.shape[0]
        self.head_size = self.embedding_dim // self.n_heads
        self.n_tokens = key.shape[-2]
        head_shape = (
            self.batch_size,
            self.n_tokens,  # number of tokens
            self.n_heads,  # number of heads
            self.head_size,  # embedding per head
        )
        out_shape = (self.batch_size, self.n_tokens, self.embedding_dim)

        # reshape and reorder the heads
        key = key._data
        query = query._data
        value = value._data

        key = key.reshape(head_shape)
        query = query.reshape(head_shape)
        value = value.reshape(head_shape)

        key = xp.einsum("BTNH->BNTH", key)
        query = xp.einsum("BTNH->BNTH", query)
        value = xp.einsum("BTNH->BNTH", value)

        self._key = key
        self._query = query
        self._value = value

        # attend
        self.divisor = sqrt(self.head_size)
        attention = xp.einsum("BNIh, BNJh -> BNIJ", query, key)
        attention = attention / self.divisor

        # mask
        attention = xp.where(
            self.mask[:, : self.n_tokens, : self.n_tokens], -xp.inf, attention
        )

        # softmax
        exp = xp.exp(attention - xp.max(attention, axis=-1, keepdims=True))
        denominator = xp.sum(exp, axis=-1, keepdims=True)
        attention = exp / denominator

        # dropout
        if self.dropout_prob:
            self._dropout_mask = xp.random.binomial(
                n=1, p=self.dropout_prob, size=attention.shape
            ).astype(bool)
            attention = xp.where(self._dropout_mask, 0, attention)

        # smush the heads back together
        self._before_smush = attention
        attention = xp.einsum("BNIj, BNjH -> BINH", attention, value)
        attention = attention.reshape(out_shape)

        result = to_tensor(attention, is_vector=True)
        result.back_fns = (self.backward,)
        result.args = (self._input,)
        return result

    def to_gpu(self, device: int):
        import cupy as cp

        cp.cuda.Device(device).use()
        self.mask = cp.array(self.mask)

    def from_gpu(self):
        self._mask = cp.asnumpy(self._mask)


class AttentionV2(Op):
    BLOCK_SIZE = 32

    def __init__(
        self,
        embedding_dim: int,
        n_heads: int,
        dropout_prob: float,
        context_window: int,
        device: int | str = "cpu",
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads
        self.context_window = context_window
        self.dropout_prob = dropout_prob
        self.dropout = Dropout(dropout_prob)
        self.mask = build_mask(
            context_window=self.context_window, n_heads=self.n_heads
        )

        self._pre_attention = None
        self._out = None
        self._attention = None

        self._load_kernels()

    def _load_kernels(self):
        include_dirs = [
            "/home/ben/mambaforge/envs/tricycle/include",  # CUDA headers
            "/usr/include",  # System headers
            "/home/ben/Documents/Tricycle/src/tricycle/cuda",
            "/home/ben/Documents/Tricycle/llm.c/dev/cuda",
            "/home/ben/mambaforge/envs/tricycle/x86-conda-linux-gnu",
            "/usr/include/x86_64-linux-gnu",
        ]
        compile_options = [
            "-DENABLE_CUDNN",
            "-lcudnn",
            "-O3",
            # "-lcublas",
            # "-lcublasLt",
            "--std=c++11",
            "-Xcompiler",
            "-fPIC",
        ]
        # load forward attention
        with open(
            "/home/ben/Documents/Tricycle/src/tricycle/cuda/attention.cu"
        ) as f:
            source = f.read()

        options = [f"-I{include_dir}" for include_dir in include_dirs]
        options.extend(compile_options)

        options = tuple(options)

        module = cp.RawModule(
            code=source,
            options=options,
            backend="nvcc",
            enable_cooperative_groups=True,
        )
        self._attention_query_key_kernel1 = module.get_function(
            "attention_query_key_kernel1"
        )
        self._attention_softmax_kernel1 = module.get_function(
            "attention_softmax_kernel1"
        )
        self._attention_value_kernel1 = module.get_function(
            "attention_value_kernel1"
        )

    def backward(self, grad: Tensor):
        return grad

    def forward(self, tensor: Tensor):
        xp = tensor.xp
        self.batch_size = tensor._data.shape[0]

        if self._out is None:
            self._out = xp.zeros(
                (self.batch_size, self.context_window, self.embedding_dim),
                dtype=tensor._data.dtype,
            )
            self._pre_attention = xp.zeros(
                (
                    self.batch_size,
                    self.n_heads,
                    self.context_window,
                    self.context_window,
                ),
                dtype=tensor._data.dtype,
            )
            self._attention = xp.zeros(
                (
                    self.batch_size,
                    self.n_heads,
                    self.context_window,
                    self.context_window,
                ),
                dtype=tensor._data.dtype,
            )

        self._input = tensor._data

        # attention
        total_threads = (
            self.batch_size
            * self.n_heads
            * self.context_window
            * self.context_window
        )
        num_blocks = 1 + (total_threads // self.BLOCK_SIZE)

        args = (
            self._pre_attention,
            self._input,
            self.batch_size,
            self.context_window,
            self.embedding_dim,
            self.n_heads,
        )
        try:
            self._attention_query_key_kernel1(
                (num_blocks,), (self.BLOCK_SIZE,), args
            )
        except Exception as e:
            breakpoint()
            raise e

        # softmax and value
        total_threads = self.batch_size * self.n_heads * self.context_window
        num_blocks = 1 + (total_threads // self.BLOCK_SIZE)

        args = (
            self._attention,
            self._pre_attention,
            self.batch_size,
            self.context_window,
            self.n_heads,
        )
        self._attention_softmax_kernel1(
            (num_blocks,), (self.BLOCK_SIZE,), args
        )
        args = (
            self._out,
            self._attention,
            self._input,
            self.batch_size,
            self.context_window,
            self.embedding_dim,
            self.n_heads,
        )
        self._attention_value_kernel1((num_blocks,), (self.BLOCK_SIZE,), args)

        return to_tensor(self._out, is_vector=tensor.is_vector)
