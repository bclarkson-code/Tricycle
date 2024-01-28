from tricycle.layers import Layer
from tricycle.tensor import Tensor
from tricycle.unary import umax


class ReLU(Layer):
    def forward(self, x: Tensor):
        return umax(x, 0)

    def update(self, _: float):
        pass

    def zero_grad(self):
        pass

    def vectorise(self):
        return self
