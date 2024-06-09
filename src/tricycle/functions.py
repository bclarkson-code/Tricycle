from tricycle.binary import BinaryDivide
from tricycle.ops import Op
from tricycle.tensor import Tensor, to_tensor
from tricycle.unary import UnaryDivide, UnaryExp


class Softmax(Op):
    def backward(self, grad: Tensor) -> Tensor:
        xp = grad.xp

        inner = xp.sum(grad.array * self._out, axis=-1, keepdims=True)
        self._grad = self._out * (grad.array - inner)
        return to_tensor(
            self._grad,
            is_batched=grad.is_batched,
            requires_grad=grad.requires_grad,
        )

    def forward(self, tensor: Tensor):
        """
        Apply softmax. The softmax is only applied to the final
        dimension of the tensor
        Note: the tensor is normalised for numeric stability
        """
        xp = tensor.xp

        exp = xp.exp(
            # subtract the largest value for numeric stability
            tensor.array
            - xp.max(tensor.array, axis=-1, keepdims=True)
        )
        denominator = xp.sum(exp, axis=-1, keepdims=True)
        self._out = exp / denominator

        result = to_tensor(self._out)
        result.args = (tensor,)
        result.name = "softmax"
        result.is_batched = tensor.is_batched
        result.back_fns = (self.backward,)

        return result


def sigmoid(tensor: Tensor):
    """
    Apply the sigmoid function
    """
    return UnaryDivide()(1, (UnaryExp()(-tensor) + 1))


def tanh(tensor: Tensor):
    """
    Apply the tanh function
    """
    numerator = UnaryExp()(tensor * 2) - 1
    denominator = UnaryExp()(tensor * 2) + 1
    return BinaryDivide()(numerator, denominator)
