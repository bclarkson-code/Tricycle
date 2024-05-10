from tricycle.functions import sigmoid
from tricycle.initialisers import init_xavier
from tricycle.layers import Dense, Layer
from tricycle.optimisers import Optimiser
from tricycle.tensor import Tensor, to_tensor
from tricycle.unary import UnaryMax


class ReLU(Layer):
    def forward(self, x: Tensor):
        return UnaryMax()(x, 0)


class Swish(Layer):
    """
    A Swish activation function. Note, because we have omitted the bias, this
    is equivalent to the Silu activation function
    """

    def forward(self, x: Tensor):
        return x * sigmoid(x)


class GeLU(Layer):
    """
    A GeLU activation function.

    Because the 100% accurate version uses erf, which involves an integral,
    we provide a fast approximation of the function
    """

    CONST_1 = 0.7978845608028654
    CONST_2 = 0.044715

    def __init__(self, *args, approximate: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.approximate = approximate

    def build_backward(self, x: Tensor):
        x = x._data

        def backward(grad: Tensor):
            xp = grad.xp

            inner = self.CONST_1 * x * (1 + self.CONST_2 * x**2)
            coef = self.CONST_1 * x * (1 + self.CONST_2 * 3 * x**2)

            left = xp.tanh(inner)
            cosh = xp.cosh(inner)
            right = coef / (cosh * cosh)
            result = 0.5 * (1 + left + right) * grad._data

            result = to_tensor(
                result,
                is_vector=grad.is_vector,
                requires_grad=grad.requires_grad,
            )
            result.name = "gelu_back"
            return result

        return backward

    def forward(self, x: Tensor):
        xp = x.xp
        inner = self.CONST_1 * (x._data + self.CONST_2 * x._data**3)
        result = x._data * 0.5 * (1 + xp.tanh(inner))

        result = to_tensor(
            result, is_vector=x.is_vector, requires_grad=x.requires_grad
        )
        result.name = "gelu"
        result.args = (x,)
        backward = self.build_backward(x)
        result.back_fns = (backward,)
        return result


class GLU(Layer):
    """
    A gated linear unit
    """

    linear: Dense

    def __init__(self, size: int, initialiser=init_xavier, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.linear = Dense(size, 2 * size, initialiser)
        self.layers = [self.linear]

    def forward(self, x: Tensor):
        x = self.linear(x)
        left, right = x.split(2)
        return left * sigmoid(right)

    def update(self, optimiser: Optimiser):
        self.linear.update(optimiser)

    def zero_grad(self):
        self.linear.zero_grad()

    def to_gpu(self):
        self.linear.to_gpu()

    def from_gpu(self):
        self.linear.from_gpu()


class SwiGLU(Layer):
    """
    A SwiGLU layer. This is a modification to a GLU where we replace
    the sigmoid with a swish
    """

    linear: Dense
    bias: Tensor

    def __init__(
        self,
        size: int,
        initialiser=init_xavier,
        tunable_bias=True,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.bias = to_tensor(1.0, requires_grad=tunable_bias, name="bias")
        self.tunable_bias = tunable_bias
        self.linear = Dense(size, 2 * size, initialiser)
        self.layers = [self.linear]

    def forward(self, x: Tensor):
        x = self.linear(x)
        # this is slow and terrible hack
        left, right = x.split(2)
        if right.is_vector:
            bias = self.bias.repeat(right.shape[1])
        else:
            bias = self.bias.repeat(right.shape[0])
        return left * (right * sigmoid(right * bias))

    def update(self, optimiser: Optimiser):
        self.linear.update(optimiser)
        self.bias = optimiser(self.bias)

    def zero_grad(self):
        self.linear.zero_grad()

    def to_gpu(self):
        self.linear.to_gpu()

    def from_gpu(self):
        self.linear.from_gpu()
