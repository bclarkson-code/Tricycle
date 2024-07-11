from abc import ABC, abstractmethod
from typing import Sequence

from numpy._typing import ArrayLike

from tricycle import TRICYCLE_CONTEXT
from tricycle.binary import BinaryMultiply
from tricycle.initialisers import init_xavier
from tricycle.optimisers import Optimiser
from tricycle.tensor import Tensor, to_tensor
from tricycle.unary import nothing


class Layer(ABC):
    """
    A generic Layer object, representing a single operation in a neural network
    """

    tensors: dict[str, Tensor] = {}
    layers: Sequence["Layer"] = []

    @abstractmethod
    def forward(self, tensor: Tensor):
        raise NotImplementedError

    def __call__(self, tensor: Tensor, *args, **kwargs):
        return self.forward(tensor, *args, **kwargs)

    def update(self, optimiser: Optimiser):
        pass

    def zero_grad(self):
        pass

    def to_gpu(self, device: int = 0):
        pass

    def from_gpu(self):
        pass


class Dense(Layer):
    weights: Tensor
    from_size: int
    to_size: int
    name: str | None

    def __init__(
        self, from_size: int, to_size: int, initialiser=init_xavier, name=None
    ):
        self.weights = initialiser(
            (from_size, to_size), name="weights" if name is None else name
        )
        self.name = name
        self.from_size = from_size
        self.to_size = to_size
        self.tensors = {"weights": self.weights}

    def weight_back_fn(self, grad: Tensor):
        xp = grad.xp

        indices = list(range(grad.ndim - 1))
        result = xp.tensordot(self._input, grad.array, axes=[indices, indices])
        return Tensor(
            result,
            requires_grad=grad.requires_grad,
            name="back_dense_weight",
            is_batched=False,
        )

    def grad_back_fn(self, grad: Tensor):
        xp = grad.xp
        result = xp.tensordot(grad.array, self.weights.array, axes=[-1, -1])
        return Tensor(
            result,
            requires_grad=grad.requires_grad,
            name="back_dense_grad",
            is_batched=True,
        )

    def forward(self, tensor: Tensor):
        xp = tensor.xp

        self._input = tensor.array
        result = xp.tensordot(tensor.array, self.weights.array, axes=[-1, 0])

        return Tensor(
            result,
            name="dense",
            args=(self.weights, tensor),
            back_fns=(self.weight_back_fn, self.grad_back_fn),
            is_batched=tensor.is_batched,
        )

    def update(self, optimiser: Optimiser):
        self.weights = optimiser(self.weights)

    def zero_grad(self):
        self.weights.grad = None

    def to_gpu(self, device: int = 0):
        self.weights.to_gpu(device)
        return self

    def from_gpu(self):
        self.weights.from_gpu()
        return self


class Dropout(Layer):
    def __init__(self, probability: float):
        self.probability = probability

    def forward(self, tensor: Tensor):
        if self.probability == 0:
            return tensor
        xp = tensor.xp
        coef = 1 / (1 - self.probability)
        random_mask = (
            xp.random.rand(*tensor.shape) > self.probability
        ).astype(tensor.dtype) * coef
        random_mask = to_tensor(
            random_mask, is_batched=True, requires_grad=False
        )
        return BinaryMultiply()(tensor, random_mask)


class LayerNorm(Layer):
    def __init__(self, embedding_dim: int, eps=1e-5):
        import numpy as np

        self.eps = eps
        self.gamma = to_tensor(
            np.ones((embedding_dim,)), requires_grad=True, is_batched=False
        )
        self.beta = to_tensor(
            np.zeros((embedding_dim,)), requires_grad=True, is_batched=False
        )

    def forward(self, tensor: Tensor):
        """
        Performs Layer Normalization on the input tensor x.

        Args:
            x (numpy.ndarray): Input tensor of shape (batch_size, *).

        Returns:
            numpy.ndarray: Normalized tensor of the same shape as x.
        """
        xp = tensor.xp
        x = tensor.array

        # Compute mean and variance along the feature dimension
        # This is pretty sensitive to errors so we need to do it at full
        # precision
        if TRICYCLE_CONTEXT.use_mixed_precision:
            x = x.astype(xp.float32)
        self._mean = x.mean(axis=-1, keepdims=True)
        self._var = x.var(axis=-1, keepdims=True)
        self._input = x

        # Normalize and scale
        x_norm = (x - self._mean) / xp.sqrt(self._var + self.eps)
        output = self.gamma.array * x_norm + self.beta.array

        if TRICYCLE_CONTEXT.use_mixed_precision:
            output = output.astype(xp.float16)

        return Tensor(
            output,
            is_batched=tensor.is_batched,
            requires_grad=tensor.requires_grad,
            back_fns=(self.back_fn, self.beta_back_fn, self.gamma_back_fn),
            args=(tensor, self.beta, self.gamma),
            name="layer_norm",
        )

    def gamma_back_fn(self, grad: Tensor):
        """
        backward pass for grad
        """
        xp = grad.xp

        # Compute intermediate values
        x_norm = (self._input - self._mean) / xp.sqrt(self._var + self.eps)
        axes = tuple(range(grad.ndim - 1))
        result = xp.sum(grad.array * x_norm, axis=axes)

        if TRICYCLE_CONTEXT.use_mixed_precision:
            result = result.astype(xp.float16)
        return Tensor(result, is_batched=False)

    def beta_back_fn(self, grad: Tensor):
        """
        backward pass for grad
        """
        xp = grad.xp

        # Compute intermediate values
        axes = tuple(range(grad.ndim - 1))
        result = xp.sum(grad.array, axis=axes)
        if TRICYCLE_CONTEXT.use_mixed_precision:
            result = result.astype(xp.float16)
        return Tensor(result, is_batched=False)

    def back_fn(self, grad: Tensor):
        """
        backward pass for grad
        """
        xp = grad.xp

        # Compute intermediate values
        n = self._input.shape[-1]

        # Gradients with respect to x
        dx_norm = grad.array * self.gamma.array
        dvar = xp.sum(
            dx_norm
            * (self._input - self._mean)
            * -0.5
            * xp.power(self._var + self.eps, -1.5),
            axis=-1,
            keepdims=True,
        )
        dmean = xp.sum(
            dx_norm * -1 / xp.sqrt(self._var + self.eps),
            axis=-1,
            keepdims=True,
        ) + dvar * xp.mean(
            -2 * (self._input - self._mean) / n, axis=-1, keepdims=True
        )
        result = (
            dx_norm / xp.sqrt(self._var + self.eps)
            + dvar * 2 * (self._input - self._mean) / n
            + dmean / n
        )

        if TRICYCLE_CONTEXT.use_mixed_precision:
            result = result.astype(xp.float16)
        return Tensor(
            result,
            is_batched=grad.is_batched,
            requires_grad=grad.requires_grad,
            name="back_ln",
        )

    def update(self, optimiser: Optimiser):
        self.gamma = optimiser(self.gamma)
        self.beta = optimiser(self.beta)

    def zero_grad(self):
        self.gamma.grad = None
        self.beta.grad = None

    def to_gpu(self, device: int = 0):
        self.gamma.to_gpu(device)
        self.beta.to_gpu(device)
        return self

    def from_gpu(self):
        self.gamma.from_gpu()
        self.beta.from_gpu()
        return self


class RMSNorm(Layer):
    def __init__(self, embedding_dim: int, REALLY_SMALL_NUMBER=1e-4):
        import numpy as np

        self.REALLY_SMALL_NUMBER = REALLY_SMALL_NUMBER
        self.embedding_dim = embedding_dim
        self.weights = Tensor(
            np.ones((embedding_dim,)), requires_grad=True, is_batched=False
        )

    def forward(self, tensor: Tensor):
        """
        Performs Layer Normalization on the input tensor x.

        Args:
            x (numpy.ndarray): Input tensor of shape (batch_size, *).

        Returns:
            numpy.ndarray: Normalized tensor of the same shape as x.
        """
        xp = tensor.xp
        x = tensor.array

        # RMSNorm is pretty sensitive to errors so we'll use full precision
        if TRICYCLE_CONTEXT.use_mixed_precision:
            x = x.astype(xp.float32)
            self.weights.array = self.weights.array.astype(xp.float32)

        # breakpoint()
        # Compute square mean along the feature dimension
        mean_square = (x**2).mean(axis=-1, keepdims=True)
        self._input = x

        # Rescale
        self._divisor = 1 / xp.sqrt(mean_square) + self.REALLY_SMALL_NUMBER
        x_norm = x * self._divisor
        output = self.weights.array * x_norm

        if TRICYCLE_CONTEXT.use_mixed_precision:
            output = output.astype(xp.float16)

        return Tensor(
            output,
            is_batched=tensor.is_batched,
            requires_grad=tensor.requires_grad,
            back_fns=(self.back_fn, self.weight_back_fn),
            args=(tensor, self.weights),
            name="rms_norm",
        )

    def weight_back_fn(self, grad: Tensor):
        """
        backward pass for grad
        """
        xp = grad.xp

        # Compute intermediate values
        # We could have stored this but I've opted for saving memory by
        # recomputing

        x_norm = self._input * self._divisor
        axes = tuple(range(grad.ndim - 1))
        result = xp.sum(grad.array * x_norm, axis=axes)

        if TRICYCLE_CONTEXT.use_mixed_precision:
            result = result.astype(xp.float16)
        return Tensor(result, is_batched=False)

    def back_fn(self, grad: Tensor):
        """
        backward pass for grad
        """
        xp = grad.xp

        left = grad.array * self.weights.array

        right = self._input * self.weights.array
        right *= (self._divisor**2) / self.embedding_dim
        right *= xp.sum(self._input * grad.array, axis=-1, keepdims=True)

        result = left - right
        result *= self._divisor

        if TRICYCLE_CONTEXT.use_mixed_precision:
            result = result.astype(xp.float16)
        return Tensor(
            result,
            is_batched=grad.is_batched,
            requires_grad=grad.requires_grad,
            name="back_rms_norm",
        )

    def update(self, optimiser: Optimiser):
        self.weights = optimiser(self.weights)

    def zero_grad(self):
        self.weights.grad = None

    def to_gpu(self, device: int = 0):
        self.weights.to_gpu(device)
        return self

    def from_gpu(self):
        self.weights.from_gpu()
        return self


class Embedding(Layer):
    """
    Convert an index to an embedding with a lookup (rather than a one-hot
    encoding and a matrix multiplication)
    """

    def __init__(
        self,
        from_size: int,
        to_size: int,
        name: str | None = None,
        initialiser=init_xavier,
    ):
        self.weights = initialiser(
            (from_size, to_size), name=name or "weights"
        )
        self.vocab_size = from_size

    def back_fn(self, grad: Tensor):
        xp = grad.xp
        out = xp.zeros(self.weights.shape)

        match grad.ndim - self.input.ndim:
            case 1:
                xp.add.at(out, self.input.array, grad.array)
            case 2:
                xp.add.at(out, self.input.array, grad.array.sum(axis=0))
            case _:
                raise NotImplementedError(
                    f"{grad.ndim=}, {self.input.ndim=} are not supported"
                )

        return Tensor(out, requires_grad=grad.requires_grad)

    def forward(self, tensor: Tensor):
        assert (
            tensor.requires_grad is False
        ), "Cannot embed a differentiable tensor"

        self.input = tensor
        if tensor.is_batched:
            self._out = self.weights.array[tensor.array.flatten()].reshape(
                tensor.array.shape + (-1,)
            )
        else:
            self._out = self.weights.array[tensor.array]
        result = to_tensor(self._out, is_batched=tensor.is_batched)

        result.args = (tensor, self.weights)

        result.back_fns = (nothing, self.back_fn)
        return result

    def update(self, optimiser: Optimiser):
        self.weights = optimiser(self.weights)

    def zero_grad(self):
        self.weights.grad = None

    def to_gpu(self, device: int = 0):
        self.weights.to_gpu(device)
        return self

    def from_gpu(self):
        self.weights.from_gpu()
        return self


class Sequential(Layer):
    def __init__(self, *layers: Layer):
        self.layers = layers

    def __getitem__(self, idx):
        return self.layers[idx]

    def forward(self, tensor: Tensor):
        for layer in self.layers:
            tensor = layer(tensor)
        return tensor

    def update(self, optimiser: Optimiser):
        for layer in self.layers:
            layer.update(optimiser)

    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()

    def to_gpu(self, device: int = 0):
        for layer in self.layers:
            layer.to_gpu(device)

    def from_gpu(self):
        for layer in self.layers:
            layer.from_gpu()


class RotaryEncode(Layer):
    """
    Apply rotary positional encoding to a key and query
    """

    embedding_dim: int
    n_heads: int
    context_window: int
    theta: float = 10_000.0

    def __init__(
        self,
        embedding_dim: int,
        n_heads: int,
        context_window: int,
        theta: float | None = None,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads
        self.context_window = context_window

        self.head_size = self.context_window // self.n_heads

        if theta is not None:
            self.theta = theta

        self.freqs_cos, self.freqs_sin = self.precompute_constants()

    def precompute_constants(self) -> tuple[ArrayLike, ArrayLike]:
        # this is run once at initialisation so we dont get any benefit from
        # cupy using
        import numpy as np

        # [0, 1, 2, ..., (dim//2) - 1]
        head_idx = np.arange(0, self.head_size // 2)

        # [0/dim, 2/dim, 4/dim, ... (dim-4) / dim, (dim-2) / dim]
        power = 2 * head_idx / self.head_size
        freqs = 1 / (self.theta**power)

        # assign an index to each token
        token_idx = np.arange(self.context_window)

        # this is a 2d matrix
        # freqs = t / theta**(2*d / dim)
        # where t is a token index and d is a head index
        freqs = np.outer(token_idx, freqs)

        freqs_cos = np.cos(freqs)
        freqs_sin = np.sin(freqs)

        return freqs_cos, freqs_sin

    def backward(self, grad: Tensor) -> Tensor:
        xp = grad.xp

        # split the final dimension in 2 putting every
        # 2i'th value in a tensor called "grad_real"
        # and every 2i + 1'th value in a tensor called "grad_imaginary"
        grad_real = grad.array[..., 0::2]
        grad_imaginary = grad.array[..., 1::2]

        input_grad_real = (
            grad_real * self.freqs_cos + grad_imaginary * self.freqs_sin
        )
        input_grad_imaginary = (
            -grad_real * self.freqs_sin + grad_imaginary * self.freqs_cos
        )

        # Interleave the gradients back together so we get:
        # real, imaginary, real, imaginary, ...
        out = xp.empty(grad.shape)
        out[..., 0::2] = input_grad_real
        out[..., 1::2] = input_grad_imaginary

        return Tensor(
            array=out,
            name="back_rotary_encode",
            is_batched=grad.is_batched,
        )

    def forward(self, tensor: Tensor) -> Tensor:
        xp = tensor.xp

        # split the final dimension in 2 putting every
        # 2i'th value an a tensor called "real"
        # and every 2i + 1'th value in a tensor called "imaginary"
        real = tensor.array[..., 0::2]
        imaginary = tensor.array[..., 1::2]

        # combine the real an imaginary parts together with frequencies
        out_real = real * self.freqs_cos - imaginary * self.freqs_sin
        out_imaginary = real * self.freqs_sin + imaginary * self.freqs_cos

        # Interleave the real and imaginary parts
        # back together so we get:
        # real, imaginary, real, imaginary, ...
        # in the final dimension
        out = xp.empty(tensor.shape)
        out[..., 0::2] = out_real
        out[..., 1::2] = out_imaginary

        return Tensor(
            array=out,
            args=(tensor,),
            back_fns=(self.backward,),
            name="rotary_encode",
            is_batched=tensor.is_batched,
        )
