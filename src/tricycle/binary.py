"""
In tricycle (because it makes the derivatives easier) we only allow operations
on two matrices if they are the same shape. We call these `binary` operations.
Because each binary Op has 2 inputs, they also need two back_fns, one for 
each input.

This file contains all of the binary operations in tricycle

In deep learning, almost all of the time you can use an einsum operation to
handle what you want to do. This includes:
 - Transposing
 - Elementwise multiplication
 - Matrix multiplication
 - ...

Interestingly, all of the operations here can be made out of clever
combinations of unary operations and einsums,  (exercise for the reader?)
but it is a bit more efficient to give them their own, optimised `Op`s
"""

from numpy.typing import ArrayLike

from tricycle.ops import Einsum, Op
from tricycle.tensor import Tensor, nothing, select_backend
from tricycle.unary import UnaryDivide
from tricycle.utils import shapes_match


class BinaryAdd(Op):
    def forward(self, tensor_1: Tensor, tensor_2: Tensor) -> Tensor:
        """
        Add two tensors together, elementwise
        """
        xp = select_backend(tensor_1.array, tensor_2.array)

        assert shapes_match(tensor_1, tensor_2)
        self._out = xp.add(tensor_1.array, tensor_2.array)

        return Tensor(
            self._out,
            args=(tensor_1, tensor_2),
            back_fns=(nothing, nothing),
            name="badd",
            is_batched=tensor_1.is_batched or tensor_2.is_batched,
        )


class BinarySubtract(Op):
    def back_fn_2(self, grad: Tensor) -> Tensor:
        self._grad = -grad.array

        return Tensor(array=self._grad, is_batched=grad.is_batched)

    def forward(self, tensor_1: Tensor, tensor_2: Tensor) -> Tensor:
        """
        Subtract one tensor from another

        The two tensors must have the same shape
        """
        xp = select_backend(tensor_1.array, tensor_2.array)

        assert shapes_match(tensor_1, tensor_2)
        self._out = xp.subtract(tensor_1.array, tensor_2.array)

        return Tensor(
            self._out,
            args=(tensor_1, tensor_2),
            back_fns=(nothing, self.back_fn_2),
            name="bsub",
            is_batched=tensor_1.is_batched or tensor_2.is_batched,
        )


class BinaryMultiply(Op):
    def forward(self, tensor_1: Tensor, tensor_2: Tensor) -> Tensor:
        """
        Multiply the elements of two tensors together, elementwise

        The two tensors must have the same shape
        """
        assert shapes_match(tensor_1, tensor_2)

        result = Einsum("...,...->...")(tensor_1, tensor_2)
        result.name = "bmul"
        return result


class BinaryDivide(Op):
    # TODO: we should probably fuse these into a single op
    def forward(self, tensor_1: Tensor, tensor_2: Tensor) -> Tensor:
        """
        Divide the elements of two tensors together, elementwise

        The two tensors must have the same shape
        """
        mul = BinaryMultiply()
        div = UnaryDivide()

        return mul(tensor_1, div(1, tensor_2))


class BinaryMax(Op):
    _is_bigger_1: ArrayLike
    _is_bigger_2: ArrayLike

    def back_fn_1(self, grad: Tensor) -> Tensor:
        self._grad_1 = grad.array * self._is_bigger_1

        return Tensor(array=self._grad_1, is_batched=grad.is_batched)

    def back_fn_2(self, grad: Tensor) -> Tensor:
        self._grad_2 = grad.array * self._is_bigger_2

        return Tensor(array=self._grad_2, is_batched=grad.is_batched)

    def forward(self, tensor_1: Tensor, tensor_2: Tensor) -> Tensor:
        """
        Compare two tensors elementwise, returning the maximum
        of each pair of elements

        The two tensors must have the same shape
        if elements are equal, return the first
        """
        xp = select_backend(tensor_1.array, tensor_2.array)
        assert shapes_match(tensor_1, tensor_2)

        self._out = xp.maximum(tensor_1.array, tensor_2.array)

        self._is_bigger_1 = tensor_1.array > tensor_2.array
        self._is_bigger_2 = tensor_1.array <= tensor_2.array

        return Tensor(
            self._out,
            args=(tensor_1, tensor_2),
            back_fns=(self.back_fn_1, self.back_fn_2),
            name="bmax",
            is_batched=tensor_1.is_batched or tensor_2.is_batched,
        )


class BinaryMin(Op):
    _is_smaller_1: ArrayLike
    _is_smaller_2: ArrayLike

    def back_fn_1(self, grad: Tensor) -> Tensor:
        self._grad_1 = grad.array * self._is_smaller_1

        return Tensor(array=self._grad_1, is_batched=grad.is_batched)

    def back_fn_2(self, grad: Tensor) -> Tensor:
        self._grad_2 = grad.array * self._is_smaller_2

        return Tensor(array=self._grad_2, is_batched=grad.is_batched)

    def forward(self, tensor_1: Tensor, tensor_2: Tensor) -> Tensor:
        """
        Compare two tensors elementwise, returning the minimum
        of each pair of elements

        The two tensors must have the same shape
        if elements are equal, return the first
        """
        xp = select_backend(tensor_1.array, tensor_2.array)
        assert shapes_match(tensor_1, tensor_2)

        self._out = xp.minimum(tensor_1.array, tensor_2.array)

        self._is_smaller_1 = tensor_1.array < tensor_2.array
        self._is_smaller_2 = tensor_1.array >= tensor_2.array

        return Tensor(
            self._out,
            args=(tensor_1, tensor_2),
            back_fns=(self.back_fn_1, self.back_fn_2),
            name="bmin",
            is_batched=tensor_1.is_batched or tensor_2.is_batched,
        )
