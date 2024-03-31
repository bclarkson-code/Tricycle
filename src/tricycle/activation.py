import numpy as np

from tricycle.functions import sigmoid, tanh
from tricycle.initialisers import init_xavier
from tricycle.layers import Dense, Layer
from tricycle.optimisers import Optimiser
from tricycle.tensor import Tensor, to_tensor
from tricycle.unary import uerf, umax


class ReLU(Layer):
    def forward(self, x: Tensor):
        return umax(x, 0)


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

    Because the default version uses erf, which involves an integral,
    we also provide a fast approximation of the function
    """

    approximate: bool

    def __init__(self, *args, approximate: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.approximate = approximate

    def forward(self, x: Tensor):
        if not self.approximate:
            SQRT_2 = 1.4142135623730951
            return x * 0.5 * (1 + uerf(x / SQRT_2))

        inner = 0.7978845608028654 * (x + 0.044715 * x**3)
        return x * 0.5 * (1 + tanh(inner))


class GLU(Layer):
    """
    A gated linear unit
    """

    linear: Dense

    def __init__(self, size: int, initialiser=init_xavier, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.linear = Dense(size, 2 * size, initialiser)

    def forward(self, x: Tensor):
        x = self.linear(x)
        left, right = x.split(2)
        return left * sigmoid(right)

    def update(self, optimiser: Optimiser):
        self.linear.update(optimiser)

    def zero_grad(self):
        self.linear.zero_grad()


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
