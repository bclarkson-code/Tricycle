from tricycle import TRICYCLE_CONTEXT
from tricycle.ops import Op
from tricycle.tensor import Tensor


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


class Sigmoid(Op):
    def backward(self, grad: Tensor) -> Tensor:
        self._grad = self._out * (1 - self._out) * grad.array
        return Tensor(self._grad, requires_grad=grad)

    def forward(self, tensor: Tensor) -> Tensor:
        """
        Apply the sigmoid function
        """
        xp = tensor.xp

        # Exponents tend to overflow/underflow when using 16 bit precision
        # so we need to switch to 32 bit
        if TRICYCLE_CONTEXT.use_mixed_precision:
            tensor.array = tensor.array.astype(xp.float32)

        self._out = 1 / (1 + xp.exp(-tensor.array))
        return Tensor(
            self._out,
            back_fns=(self.backward,),
            args=(tensor,),
            requires_grad=tensor.requires_grad,
        )
