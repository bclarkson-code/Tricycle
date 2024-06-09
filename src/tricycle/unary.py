"""
When doing tensor calculus, some Operations have a single input and output.
I'm calling these `unary` operations.

This file contains all of the unary operations in Tricycle
"""

import numbers

from numpy.typing import ArrayLike

from tricycle.ops import Op
from tricycle.tensor import Tensor, nothing, to_tensor


class UnaryAdd(Op):
    def forward(self, tensor: Tensor, constant: float) -> Tensor:
        """
        Add a constant, elementwise, to a tensor. The constant is not
        differentiable.
        """
        xp = tensor.xp

        assert isinstance(tensor, Tensor)
        assert isinstance(constant, numbers.Number)

        self._out = xp.add(tensor.array, constant)

        result = to_tensor(self._out)
        result.args = (tensor,)
        result.back_fns = (nothing,)
        result.name = f"+ {constant}"
        result.is_vector = tensor.is_vector
        return result


class UnaryMultiply(Op):
    _constant: float

    def back_fn(self, grad: Tensor) -> Tensor:
        xp = grad.xp

        self._grad = xp.multiply(grad.array, self._constant)
        result = to_tensor(self._grad)
        result.is_vector = grad.is_vector
        return result

    def forward(self, tensor: Tensor, constant: float) -> Tensor:
        """
        Multiply a constant, elementwise, to a tensor. The constant is not
        differentiable.
        """
        xp = tensor.xp

        assert isinstance(tensor, Tensor)
        assert xp.isscalar(constant)

        self._out = xp.multiply(tensor.array, constant)
        self._constant = constant

        result = to_tensor(self._out)
        result.args = (tensor,)
        result.back_fns = (self.back_fn,)
        result.name = f"+ {constant}"
        result.is_vector = tensor.is_vector
        return result


class UnarySubtract(Op):
    def forward(self, tensor: Tensor, constant: float) -> Tensor:
        """
        Subtract a constant, elementwise, from a tensor. The constant is not
        differentiable.
        """
        return UnaryAdd()(tensor, -constant)


class UnaryPower(Op):
    input: ArrayLike
    constant: float

    def back_fn(self, grad: Tensor) -> Tensor:
        xp = grad.xp

        self._grad = xp.power(
            self.input._data, self.constant - 1, dtype=self.input.dtype
        )
        self._grad *= self.constant * grad.array

        result = to_tensor(self._grad)
        result.is_vector = grad.is_vector
        return result

    def forward(self, tensor: Tensor, constant: float) -> Tensor:
        """
        Raise a tensor to a constant, elementwise. The constant is not
        differentiable.
        """
        xp = tensor.xp

        assert isinstance(tensor, Tensor)
        assert xp.isscalar(constant)

        self._out = xp.power(tensor.array, constant)
        self.input = tensor
        self.constant = constant

        result = to_tensor(self._out)
        result.args = (tensor,)
        result.back_fns = (self.back_fn,)
        result.name = f"^ {constant}"
        result.is_vector = tensor.is_vector

        return result


class UnaryDivide(Op):
    # TODO: manually define the derivative instead of using other operations
    def forward(self, constant: float | int, tensor: Tensor) -> Tensor:
        """
        Divide a constant by a tensor, elementwise. The constant is not
        differentiable.
        """
        upow = UnaryPower()
        umul = UnaryMultiply()
        return umul(upow(tensor, -1.0), constant)


class UnaryMax(Op):
    is_bigger: Tensor

    def back_fn(self, grad: Tensor) -> Tensor:
        self._grad = grad.array * self.is_bigger.array

        result = to_tensor(self._grad)
        result.is_vector = grad.is_vector
        return result

    def forward(self, tensor: Tensor, constant: float) -> Tensor:
        """
        Return the max of the tensor and the constant,
        elementwise. The constant is not differentiable.
        """
        xp = tensor.xp

        assert isinstance(tensor, Tensor)
        assert xp.isscalar(constant)

        self._out = xp.maximum(tensor.array, constant, dtype=tensor.dtype)

        self.is_bigger = tensor > constant
        self.is_bigger.is_vector = tensor.is_vector

        result = to_tensor(self._out)
        result.args = (tensor,)
        result.back_fns = (self.back_fn,)
        result.name = f"> {constant}"
        result.is_vector = tensor.is_vector

        return result


class UnaryMin(Op):
    is_smaller: Tensor

    def back_fn(self, grad: Tensor) -> Tensor:
        self._grad = grad.array * self.is_smaller.array

        result = to_tensor(self._grad)
        result.is_vector = grad.is_vector
        return result

    def forward(self, tensor: Tensor, constant: float) -> Tensor:
        """
        Return the max of the tensor and the constant,
        elementwise. The constant is not differentiable.
        """
        xp = tensor.xp

        assert isinstance(tensor, Tensor)
        assert xp.isscalar(constant)

        self._out = xp.minimum(tensor.array, constant, dtype=tensor.dtype)

        self.is_smaller = tensor < constant
        self.is_smaller.is_vector = tensor.is_vector

        result = to_tensor(self._out)
        result.args = (tensor,)
        result.back_fns = (self.back_fn,)
        result.name = f"> {constant}"
        result.is_vector = tensor.is_vector

        return result


class UnaryExp(Op):
    def back_fn(self, grad: Tensor) -> Tensor:
        self._grad = grad.array * self._out

        result = to_tensor(self._grad)
        result.is_vector = grad.is_vector
        return result

    def forward(self, tensor: Tensor) -> Tensor:
        """
        Raise every element of a tensor to the power of e
        """
        xp = tensor.xp

        self._out = xp.exp(tensor.array)

        result = to_tensor(self._out)
        result.args = (tensor,)
        result.back_fns = (self.back_fn,)
        result.name = "exp"
        result.is_vector = tensor.is_vector
        return result


class UnaryLog(Op):
    REALLY_SMALL_NUMBER = 1e-8

    _input: ArrayLike | None = None

    def back_fn(self, grad: Tensor) -> Tensor:
        xp = grad.xp
        denominator = self._input + self.REALLY_SMALL_NUMBER
        self._grad = grad.array * xp.divide(1, denominator)

        result = to_tensor(self._grad)
        result.is_vector = grad.is_vector
        return result

    def forward(self, tensor: Tensor) -> Tensor:
        """
        Raise every element of a tensor to the power of e
        """
        xp = tensor.xp

        self._out = xp.log(tensor.array)
        self._input = tensor.array

        result = to_tensor(self._out)

        result.args = (tensor,)
        result.back_fns = (self.back_fn,)
        result.name = "log"
        result.is_vector = tensor.is_vector
        return result


class UnarySin(Op):
    _input: ArrayLike | None = None

    def back_fn(self, grad: Tensor) -> Tensor:
        xp = grad.xp

        self._grad = grad.array * xp.cos(self._input)

        result = to_tensor(self._grad)
        result.is_vector = grad.is_vector
        return result

    def forward(self, tensor: Tensor) -> Tensor:
        """
        Applies the sine function, elementwise, to a tensor
        """
        xp = tensor.xp

        self._out = xp.sin(tensor.array)
        self._input = tensor.array

        result = to_tensor(self._out)
        result.args = (tensor,)
        result.back_fns = (self.back_fn,)
        result.name = "sin"
        result.is_vector = tensor.is_vector
        return result


class UnaryCos(Op):
    _input: ArrayLike | None = None

    def back_fn(self, grad: Tensor) -> Tensor:
        xp = grad.xp

        self._grad = grad.array * -xp.sin(self._input)

        result = to_tensor(self._grad)
        result.is_vector = grad.is_vector
        return result

    def forward(self, tensor: Tensor) -> Tensor:
        """
        Applies the cosine function, elementwise, to a tensor
        """
        xp = tensor.xp

        self._out = xp.cos(tensor.array)
        self._input = tensor.array

        result = to_tensor(self._out)
        result.args = (tensor,)
        result.back_fns = (self.back_fn,)
        result.name = "cos"
        result.is_vector = tensor.is_vector
        return result


class UnarySquareRoot(Op):
    def forward(self, tensor: Tensor):
        """
        Apply the square root function
        """
        upow = UnaryPower()
        return upow(tensor, 0.5)


class UnarySum(Op):
    _in_shape: tuple[int]
    _in_is_vector: bool

    def back_fn(self, grad: Tensor) -> Tensor:
        xp = grad.xp

        self._grad = xp.full(self._in_shape, grad.array)

        result = to_tensor(self._grad)
        result.is_vector = self._in_is_vector
        return result

    def forward(self, tensor: Tensor) -> Tensor:
        """
        Sums all the values in a tensor
        """
        xp = tensor.xp

        # Sum all the values in the tensor
        self._out = xp.sum(tensor.array)
        self._in_shape = tensor.shape
        self._in_is_vector = tensor.is_vector

        result = to_tensor(self._out)
        result.args = (tensor,)
        result.back_fns = (self.back_fn,)
        result.name = "sum"
        result.is_vector = False  # The result of the sum is a scalar
        return result
