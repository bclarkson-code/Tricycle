from tricycle.layers import Layer
from tricycle.ops import sigmoid
from tricycle.optimisers import Optimiser
from tricycle.tensor import Tensor
from tricycle.unary import umax


class ReLU(Layer):
    def forward(self, x: Tensor):
        return umax(x, 0)

    def update(self, optimiser: Optimiser):
        pass

    def zero_grad(self):
        pass


class Swish(Layer):
    """
    A Swish activation function. Note, because we have omitted the bias, this
    is equivalent to the Silu activation function
    """

    def forward(self, x: Tensor):
        return x * sigmoid(x)

    def update(self, optimiser: Optimiser):
        pass

    def zero_grad(self):
        pass
