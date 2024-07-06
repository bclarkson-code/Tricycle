from tricycle import TRICYCLE_CONTEXT
from tricycle.binary import BinaryDivide
from tricycle.ops import Op
from tricycle.tensor import Tensor
from tricycle.unary import UnaryDivide, UnaryExp


class Softmax(Op):
    def back_fn(self, grad: Tensor) -> Tensor:
        xp = grad.xp

        inner = xp.sum(grad.array * self._out, axis=-1, keepdims=True)
        self._grad = self._out * (grad.array - inner)

        return Tensor(
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

        # Exponents tend to overflow/underflow when using 16 bit precision
        # so we need to switch to 32 bit
        if TRICYCLE_CONTEXT.use_mixed_precision:
            tensor.array = tensor.array.astype(xp.float32)
        exp = xp.exp(
            # subtract the largest value for numeric stability
            tensor.array
            - xp.max(tensor.array, axis=-1, keepdims=True)
        )
        denominator = xp.sum(exp, axis=-1, keepdims=True)
        self._out = exp / denominator
        if TRICYCLE_CONTEXT.use_mixed_precision:
            self._out = self._out.astype(xp.float16)

        return Tensor(
            self._out,
            args=(tensor,),
            name="softmax",
            is_batched=tensor.is_batched,
            back_fns=(self.back_fn,),
        )


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
