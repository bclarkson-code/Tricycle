"""
The Ops that have a single input and output are called Unary Ops
This file contains all of the unary operations in Tricycle
"""

import numbers
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike

from tricycle.ops import Op
from tricycle.tensor import Tensor, nothing


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

        return Tensor(
            array=self._out,
            args=(tensor,),
            back_fns=(nothing,),
            name=f"+ {constant}",
            is_batched=tensor.is_batched,
        )


class UnaryMultiply(Op):
    _constant: float

    def back_fn(self, grad: Tensor) -> Tensor:
        xp = grad.xp

        self._grad = xp.multiply(grad.array, self._constant)

        return Tensor(array=self._grad, is_batched=grad.is_batched)

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

        return Tensor(
            array=self._out,
            args=(tensor,),
            back_fns=(self.back_fn,),
            name=f"+ {constant}",
            is_batched=tensor.is_batched,
        )


class UnarySubtract(Op):
    """
    Subtract a constant, elementwise, from a tensor. The constant is not
    differentiable.
    """

    def forward(self, tensor: Tensor, constant: float) -> Tensor:
        """
        Subtract a constant, elementwise, from a tensor. The constant is not
        differentiable.
        """
        return UnaryAdd()(tensor, -constant)


class UnaryPower(Op):
    """
    Raise a tensor to a constant, elementwise. The constant is not
    differentiable.
    """

    _input: ArrayLike
    _constant: float

    def back_fn(self, grad: Tensor) -> Tensor:
        xp = grad.xp

        self._grad = xp.power(
            self._input, self._constant - 1, dtype=self._input.dtype
        )
        self._grad *= self._constant * grad.array

        return Tensor(array=self._grad, is_batched=grad.is_batched)

    def forward(self, tensor: Tensor, constant: float) -> Tensor:
        """
        Raise a tensor to a constant, elementwise. The constant is not
        differentiable.
        """
        xp = tensor.xp

        assert isinstance(tensor, Tensor)
        assert xp.isscalar(constant)

        self._out = xp.power(tensor.array, constant)
        self._input = tensor.array
        self._constant = constant

        return Tensor(
            array=self._out,
            args=(tensor,),
            back_fns=(self.back_fn,),
            name=f"^ {constant}",
            is_batched=tensor.is_batched,
        )


class UnaryDivide(Op):
    """
    Divide a constant by a tensor, elementwise. The constant is not
    differentiable.
    """

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
    """
    Return the max of the tensor and the constant,
    elementwise. The constant is not differentiable.
    """

    _is_bigger: ArrayLike

    def back_fn(self, grad: Tensor) -> Tensor:
        self._grad = grad.array * self._is_bigger

        return Tensor(array=self._grad, is_batched=grad.is_batched)

    def forward(self, tensor: Tensor, constant: float) -> Tensor:
        """
        Return the max of the tensor and the constant,
        elementwise. The constant is not differentiable.
        """
        xp = tensor.xp

        assert isinstance(tensor, Tensor)
        assert xp.isscalar(constant)

        self._out = xp.maximum(tensor.array, constant, dtype=tensor.dtype)

        self._is_bigger = tensor.array > constant

        return Tensor(
            self._out,
            args=(tensor,),
            back_fns=(self.back_fn,),
            name=f"> {constant}",
            is_batched=tensor.is_batched,
        )


class UnaryMin(Op):
    _is_smaller: Tensor

    def back_fn(self, grad: Tensor) -> Tensor:
        self._grad = grad.array * self._is_smaller

        return Tensor(array=self._grad, is_batched=grad.is_batched)

    def forward(self, tensor: Tensor, constant: float) -> Tensor:
        """
        Return the max of the tensor and the constant,
        elementwise. The constant is not differentiable.
        """
        xp = tensor.xp

        assert isinstance(tensor, Tensor)
        assert xp.isscalar(constant)

        self._out = xp.minimum(tensor.array, constant, dtype=tensor.dtype)

        self._is_smaller = tensor.array < constant

        return Tensor(
            self._out,
            args=(tensor,),
            back_fns=(self.back_fn,),
            name=f"< {constant}",
            is_batched=tensor.is_batched,
        )


class UnaryExp(Op):
    def back_fn(self, grad: Tensor) -> Tensor:
        self._grad = grad.array * self._out

        return Tensor(array=self._grad, is_batched=grad.is_batched)

    def forward(self, tensor: Tensor) -> Tensor:
        """
        Raise every element of a tensor to the power of e
        """
        xp = tensor.xp

        self._out = xp.exp(tensor.array)

        return Tensor(
            self._out,
            args=(tensor,),
            back_fns=(self.back_fn,),
            name="exp",
            is_batched=tensor.is_batched,
        )


class UnaryLog(Op):
    REALLY_SMALL_NUMBER = 1e-8

    _input: ArrayLike

    def back_fn(self, grad: Tensor) -> Tensor:
        xp = grad.xp

        denominator = self._input + self.REALLY_SMALL_NUMBER
        self._grad = grad.array * xp.divide(1, denominator)

        return Tensor(array=self._grad, is_batched=grad.is_batched)

    def forward(self, tensor: Tensor) -> Tensor:
        """
        Raise every element of a tensor to the power of e
        """
        xp = tensor.xp

        self._out = xp.log(tensor.array)
        self._input = tensor.array

        return Tensor(
            self._out,
            args=(tensor,),
            back_fns=(self.back_fn,),
            name="log",
            is_batched=tensor.is_batched,
        )


class UnarySin(Op):
    _input: ArrayLike

    def back_fn(self, grad: Tensor) -> Tensor:
        xp = grad.xp

        self._grad = grad.array * xp.cos(self._input)

        return Tensor(array=self._grad, is_batched=grad.is_batched)

    def forward(self, tensor: Tensor) -> Tensor:
        """
        Applies the sine function, elementwise, to a tensor
        """
        xp = tensor.xp

        self._out = xp.sin(tensor.array)
        self._input = tensor.array

        return Tensor(
            self._out,
            args=(tensor,),
            back_fns=(self.back_fn,),
            name="sin",
            is_batched=tensor.is_batched,
        )


class UnaryCos(Op):
    _input: ArrayLike

    def back_fn(self, grad: Tensor) -> Tensor:
        xp = grad.xp

        self._grad = grad.array * -xp.sin(self._input)

        return Tensor(array=self._grad, is_batched=grad.is_batched)

    def forward(self, tensor: Tensor) -> Tensor:
        """
        Applies the cosine function, elementwise, to a tensor
        """
        xp = tensor.xp

        self._out = xp.cos(tensor.array)
        self._input = tensor.array

        return Tensor(
            self._out,
            args=(tensor,),
            back_fns=(self.back_fn,),
            name="cos",
            is_batched=tensor.is_batched,
        )


class UnarySquareRoot(Op):
    """
    Apply the square root function
    """

    # TODO: This would probably be faster if we use xp.sqrt instead of xp.power
    def forward(self, tensor: Tensor):
        """
        Apply the square root function
        """
        upow = UnaryPower()
        return upow(tensor, 0.5)


class UnarySum(Op):
    """
    Sum all the values in a tensor
    """

    _in_shape: Sequence[int]
    _is_batched: bool

    def back_fn(self, grad: Tensor) -> Tensor:
        xp = grad.xp

        assert xp.isscalar(grad) or grad.shape == ()

        self._grad = xp.full(self._in_shape, grad.array)

        return Tensor(array=self._grad, is_batched=grad.is_batched)

    def forward(self, tensor: Tensor) -> Tensor:
        """
        Sums all the values in a tensor.
        Note, this function always produces a non-batched output
        """
        xp = tensor.xp

        self._out = xp.sum(tensor.array)
        self._in_shape = tensor.shape
        self._is_batched = tensor.is_batched

        assert self._out is not None

        return Tensor(
            array=self._out,
            args=(tensor,),
            back_fns=(self.back_fn,),
            name="sum",
            is_batched=False,
        )
