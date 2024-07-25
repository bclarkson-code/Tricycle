from tricycle.context import TRICYCLE_CONTEXT
from tricycle.functions import Sigmoid
from tricycle.initialisers import init_xavier
from tricycle.layers import Dense, Layer
from tricycle.optimisers import Optimiser
from tricycle.tensor import Tensor
from tricycle.unary import UnaryMax


class ReLU(Layer):
    """
    Rectified Linear Unit (ReLU) activation function.

    This layer applies the ReLU function element-wise to the input tensor.
    ReLU(x) = max(0, x)
    """

    def forward(self, x: Tensor):
        """
        Apply the ReLU function to the input tensor.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after applying ReLU.
        """
        return UnaryMax()(x, 0)


class Swish(Layer):
    """
    Swish activation function.

    This layer applies the Swish function element-wise to the input tensor.
    Swish(x) = x * sigmoid(x)

    Note: This implementation is equivalent to the SiLU activation function
    as it omits the bias term.
    """

    def backward(self, grad: Tensor):
        """
        Compute the gradient of the Swish function.

        Args:
            grad (Tensor): Upstream gradient.

        Returns:
            Tensor: Gradient with respect to the input.
        """
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
        """
        Apply the Swish function to the input tensor.

        Args:
            tensor (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after applying Swish.
        """
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
    Gaussian Error Linear Unit (GELU) activation function.

    This layer applies the GELU function element-wise to the input tensor.
    GELU(x) ≈ 0.5x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))

    Args:
        approximate (bool): Whether to use the approximate version of GELU.
            Defaults to False.
    """

    CONST_1 = 0.7978845608028654
    CONST_2 = 0.044715

    def __init__(self, *args, approximate: bool = False, **kwargs):
        """
        Initialize the GELU layer.

        Args:
            approximate (bool): Whether to use the approximate version of GELU.
                Defaults to False.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.approximate = approximate

    def backward(self, grad: Tensor):
        """
        Compute the gradient of the GELU function.

        Args:
            grad (Tensor): Upstream gradient.

        Returns:
            Tensor: Gradient with respect to the input.
        """
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
        """
        Apply the GELU function to the input tensor.

        Args:
            tensor (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after applying GELU.
        """
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


class GLU(Layer):
    """
    Gated Linear Unit (GLU) activation function.

    This layer applies the GLU function to the input tensor.
    GLU(x) = x_left * sigmoid(x_right)

    Args:
        size (int): Size of the input tensor.
        initialiser (callable): Function to initialize the weights.
            Defaults to init_xavier.
    """

    linear: Dense

    def __init__(self, size: int, initialiser=init_xavier, *args, **kwargs):
        """
        Initialize the GLU layer.

        Args:
            size (int): Size of the input tensor.
            initialiser (callable): Function to initialize the weights.
                Defaults to init_xavier.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.linear = Dense(size, 2 * size, initialiser)
        self.layers = [self.linear]
        self.sigmoid = Sigmoid()

    def forward(self, x: Tensor):
        """
        Apply the GLU function to the input tensor.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after applying GLU.
        """
        x = self.linear(x)
        left, right = x.split(2)
        return left * self.sigmoid(right)

    def update(self, optimiser: Optimiser):
        """
        Update the layer parameters using the given optimizer.

        Args:
            optimiser (Optimiser): The optimizer to use for updating parameters.
        """
        self.linear.update(optimiser)

    def zero_grad(self):
        """
        Reset the gradients of the layer parameters to zero.
        """
        self.linear.zero_grad()

    def to_gpu(self):
        """
        Move the layer parameters to GPU memory.
        """
        self.linear.to_gpu()

    def from_gpu(self):
        """
        Move the layer parameters from GPU to CPU memory.
        """
        self.linear.from_gpu()
