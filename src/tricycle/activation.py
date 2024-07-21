from numpy.typing import ArrayLike

from tricycle import TRICYCLE_CONTEXT
from tricycle.functions import Sigmoid
from tricycle.initialisers import init_xavier
from tricycle.layers import CudaLayer, Dense, Layer
from tricycle.optimisers import Optimiser
from tricycle.tensor import Tensor
from tricycle.unary import UnaryMax


class ReLU(Layer):
    def forward(self, x: Tensor):
        return UnaryMax()(x, 0)


class CudaReLU(CudaLayer):
    """
    Relu layer that uses a handwritten set of CUDA kernels instead of using the
    CUPY python API
    """

    def __init__(self, filename: str = "relu.cu"):
        super().__init__(filename=filename)
        self.forward_kernel = self.module.get_function("relu_forward")
        self.backward_kernel = self.module.get_function("relu_backward")

        self.below_zero = None

    def backward(self, grad: Tensor) -> Tensor:
        import cupy as cp

        output = cp.empty_like(grad.array)
        cuda_kwargs = self._calculate_cuda_block_size(grad)

        # run kernel
        self.backward_kernel(
            **cuda_kwargs,
            args=(grad.array, output, self.below_zero),
        )
        return Tensor(output, is_batched=grad.is_batched)

    def forward(self, x: Tensor):
        import cupy as cp

        if self.below_zero is None:
            self.below_zero = cp.empty(x.shape, dtype=cp.bool_)

        output = cp.empty_like(x.array)
        cuda_kwargs = self._calculate_cuda_block_size(x)

        self.forward_kernel(
            **cuda_kwargs,
            args=(x.array, output, self.below_zero),
        )
        return Tensor(
            output,
            args=(x,),
            back_fns=(self.backward,),
            is_batched=x.is_batched,
        )


class Swish(Layer):
    """
    A Swish activation function. Note, because we have omitted the bias, this
    is equivalent to the Silu activation function
    """

    def backward(self, grad: Tensor):
        xp = grad.xp

        # Exponents tend to overflow/underflow when using 16 bit precision
        # so we need to switch to 32 bit
        if TRICYCLE_CONTEXT.use_mixed_precision:
            self._input = self._input.astype(xp.float32)

        exp = xp.exp(-self._input)
        numerator = 1 + exp + self._input * exp
        denominator = (1 + exp) ** 2
        coef = numerator / denominator

        if TRICYCLE_CONTEXT.use_mixed_precision:
            coef = coef.astype(xp.float16)

        return Tensor(grad * coef)

    def forward(self, tensor: Tensor):
        xp = tensor.xp

        self._input = tensor.array
        # Exponents tend to overflow/underflow when using 16 bit precision
        # so we need to switch to 32 bit
        if TRICYCLE_CONTEXT.use_mixed_precision:
            self._input = self._input.astype(xp.float32)
        out = tensor.array / (1 + xp.exp(-tensor.array))

        if TRICYCLE_CONTEXT.use_mixed_precision:
            self._input = self._input.astype(xp.float16)
            out = out.astype(xp.float16)

        return Tensor(
            out, args=(tensor,), back_fns=(self.backward,), name="swish"
        )


class GeLU(Layer):
    """
    A GeLU activation function.

    Because the 100% accurate version uses erf, which involves an integral,
    we're using a fast approximation of the function from OpenAI
    """

    CONST_1 = 0.7978845608028654
    CONST_2 = 0.044715

    def __init__(self, *args, approximate: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.approximate = approximate

    def backward(self, grad: Tensor):
        xp = grad.xp

        # Hyperbolic trig functions (cosh and tanh) use exponents under the
        # hood which can overflow/underflow when using 16 bit precision so
        # we need to switch to 32 bit precision
        if TRICYCLE_CONTEXT.use_mixed_precision:
            self._input = self._input.astype(xp.float32)

        inner = (
            self.CONST_1 * self._input * (1 + self.CONST_2 * self._input**2)
        )
        coef = (
            self.CONST_1
            * self._input
            * (1 + self.CONST_2 * 3 * self._input**2)
        )

        left = xp.tanh(inner)
        cosh = xp.cosh(inner)
        right = coef / (cosh * cosh)

        if TRICYCLE_CONTEXT.use_mixed_precision:
            left = left.astype(xp.float16)
            right = right.astype(xp.float16)

        self._grad = 0.5 * (1 + left + right) * grad.array

        result = Tensor(
            self._grad,
            is_batched=grad.is_batched,
            requires_grad=grad.requires_grad,
        )
        result.name = "gelu_back"
        return result

    def forward(self, tensor: Tensor):
        xp = tensor.xp
        self._input = tensor.array

        # Tanh tends to overflow/underflow when using 16 bit precision
        # so we need to switch to 32 bit
        if TRICYCLE_CONTEXT.use_mixed_precision:
            self._input = self._input.astype(xp.float32)

        inner = self.CONST_1 * (self._input + self.CONST_2 * self._input**3)
        result = self._input * 0.5 * (1 + xp.tanh(inner))

        if TRICYCLE_CONTEXT.use_mixed_precision:
            self._input = self._input.astype(xp.float16)
            result = result.astype(xp.float16)

        result = Tensor(
            result,
            is_batched=tensor.is_batched,
            requires_grad=tensor.requires_grad,
        )
        result.name = "gelu"
        result.args = (tensor,)
        result.back_fns = (self.backward,)
        return result


class CudaGeLU(CudaLayer):
    """
    Gelu layer that uses a handwritten set of CUDA kernels instead of using the
    CUPY python API
    """

    def __init__(self, filename: str = "gelu.cu"):
        super().__init__(filename=filename)
        self.forward_kernel = self.module.get_function("gelu_forward")
        self.backward_kernel = self.module.get_function("gelu_backward")

        # preallcate an output buffer
        self.output = None

    def backward(self, grad: Tensor) -> Tensor:
        cuda_kwargs = self._calculate_cuda_block_size(grad)

        # run kernel
        # note: the gradient update is done inplace
        self.backward_kernel(
            **cuda_kwargs,
            args=(grad.array, self.input),
        )
        return Tensor(grad.array, is_batched=grad.is_batched)

    def forward(self, tensor: Tensor):
        import cupy as cp

        self.input = tensor.array

        output = cp.empty_like(self.input)

        cuda_kwargs = self._calculate_cuda_block_size(tensor)

        self.forward_kernel(
            **cuda_kwargs,
            args=(output, self.input),
        )
        return Tensor(
            output,
            args=(tensor,),
            back_fns=(self.backward,),
            is_batched=tensor.is_batched,
        )


class GLU(Layer):
    """
    A gated linear unit
    """

    linear: Dense

    def __init__(self, size: int, initialiser=init_xavier, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.linear = Dense(size, 2 * size, initialiser)
        self.layers = [self.linear]
        self.sigmoid = Sigmoid()

    def forward(self, x: Tensor):
        x = self.linear(x)
        left, right = x.split(2)
        return left * self.sigmoid(right)

    def update(self, optimiser: Optimiser):
        self.linear.update(optimiser)

    def zero_grad(self):
        self.linear.zero_grad()

    def to_gpu(self):
        self.linear.to_gpu()

    def from_gpu(self):
        self.linear.from_gpu()
