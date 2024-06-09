from functools import partial

from numpy.typing import ArrayLike

from tricycle.ops import Einsum, Op
from tricycle.tensor import Tensor, nothing, select_backend, to_tensor
from tricycle.unary import UnaryDivide, UnaryMultiply


def _shapes_match(tensor_1: Tensor, tensor_2: Tensor) -> bool:
    # sourcery skip: assign-if-exp, merge-duplicate-blocks, remove-redundant-if
    if tensor_1.is_vector and tensor_2.is_vector:
        shape_1 = tensor_1.shape
        shape_2 = tensor_2.shape
    elif tensor_1.is_vector:
        shape_1 = tensor_1.shape[1:]
        shape_2 = tensor_2.shape
    elif tensor_2.is_vector:
        shape_1 = tensor_1.shape
        shape_2 = tensor_2.shape[1:]
    else:
        shape_1 = tensor_1.shape
        shape_2 = tensor_2.shape

    if shape_1 != shape_2:
        raise ValueError(
            f"Shapes {shape_1} and {shape_2} do not match: {tensor_1.array.shape}, {tensor_2.array.shape}"
        )
    return shape_1 == shape_2


class BinaryAdd(Op):
    def forward(self, tensor_1: Tensor, tensor_2: Tensor) -> Tensor:
        """
        Applies the cosine function, elementwise, to a tensor
        """
        xp = select_backend(tensor_1.array, tensor_2.array)

        assert _shapes_match(tensor_1, tensor_2)
        self._out = xp.add(tensor_1.array, tensor_2.array)

        result = to_tensor(self._out)
        result.args = (tensor_1, tensor_2)
        result.back_fns = (nothing, nothing)
        result.name = "badd"

        if tensor_1.is_vector or tensor_2.is_vector:
            result.is_vector = True

        return result


class BinarySubtract(Op):
    def back_fn_2(self, grad: Tensor) -> Tensor:
        self._grad = -grad.array
        result = to_tensor(self._grad)
        result.is_vector = grad.is_vector
        return result

    def forward(self, tensor_1: Tensor, tensor_2: Tensor) -> Tensor:
        """
        Subtract one tensor from another
        """
        xp = select_backend(tensor_1.array, tensor_2.array)

        assert _shapes_match(tensor_1, tensor_2)
        self._out = xp.subtract(tensor_1.array, tensor_2.array)

        result = to_tensor(self._out)
        result.args = (tensor_1, tensor_2)
        result.back_fns = (nothing, self.back_fn_2)
        result.name = "badd"

        if tensor_1.is_vector or tensor_2.is_vector:
            result.is_vector = True

        return result


class BinaryMultiply(Op):
    def forward(self, tensor_1: Tensor, tensor_2: Tensor) -> Tensor:
        """
        Multiply the elements of two tensors together, elementwise

        The two tensors must have the same shape
        """
        assert _shapes_match(tensor_1, tensor_2)

        result = Einsum("...,...->...")(tensor_1, tensor_2)
        result.name = "bmul"
        return result


class BinaryDivide(Op):
    def forward(self, tensor_1: Tensor, tensor_2: Tensor) -> Tensor:
        """
        Divide the elements of two tensors together, elementwise

        The two tensors must have the same shape
        """
        mul = BinaryMultiply()
        div = UnaryDivide()

        return mul(tensor_1, div(1, tensor_2))


class BinaryMax(Op):
    _is_bigger_1: ArrayLike | None
    _is_bigger_2: ArrayLike | None

    def back_fn_1(self, grad: Tensor) -> Tensor:
        self._grad_1 = grad.array * self._is_bigger_1
        result = to_tensor(self._grad_1)
        result.is_vector = grad.is_vector
        return result

    def back_fn_2(self, grad: Tensor) -> Tensor:
        self._grad_2 = grad.array * self._is_bigger_2
        result = to_tensor(self._grad_2)
        result.is_vector = grad.is_vector
        return result

    def forward(self, tensor_1: Tensor, tensor_2: Tensor) -> Tensor:
        """
        Compare two tensors elementwise, returning the maximum
        of each pair of elements

        The two tensors must have the same shape
        if elements are equal, return the first
        """
        xp = select_backend(tensor_1.array, tensor_2.array)
        assert _shapes_match(tensor_1, tensor_2)

        self._out = xp.maximum(tensor_1.array, tensor_2.array)

        self._is_bigger_1 = tensor_1.array > tensor_2.array
        self._is_bigger_2 = tensor_1.array <= tensor_2.array

        result = to_tensor(self._out)
        result.args = (tensor_1, tensor_2)
        result.back_fns = (self.back_fn_1, self.back_fn_2)
        result.name = "bmax"
        result.is_vector = tensor_1.is_vector or tensor_2.is_vector
        return result


class BinaryMin(Op):
    _is_smaller_1: Tensor | None
    _is_smaller_2: Tensor | None

    def back_fn_1(self, grad: Tensor) -> Tensor:
        self._grad_1 = grad.array * self._is_smaller_1
        result = to_tensor(self._grad_1)
        result.is_vector = grad.is_vector
        return result

    def back_fn_2(self, grad: Tensor) -> Tensor:
        self._grad_2 = grad.array * self._is_smaller_2
        result = to_tensor(self._grad_2)
        result.is_vector = grad.is_vector
        return result

    def forward(self, tensor_1: Tensor, tensor_2: Tensor) -> Tensor:
        """
        Compare two tensors elementwise, returning the maximum
        of each pair of elements

        The two tensors must have the same shape
        if elements are equal, return the first
        """
        xp = select_backend(tensor_1.array, tensor_2.array)
        assert _shapes_match(tensor_1, tensor_2)

        self._out = xp.minimum(tensor_1.array, tensor_2.array)

        self._is_smaller_1 = tensor_1.array < tensor_2.array
        self._is_smaller_2 = tensor_1.array >= tensor_2.array

        result = to_tensor(self._out)
        result.args = (tensor_1, tensor_2)
        result.back_fns = (self.back_fn_1, self.back_fn_2)
        result.name = "bmax"
        result.is_vector = tensor_1.is_vector or tensor_2.is_vector
        return result


class BinaryMask(Op):
    _mask: ArrayLike | None = None

    def back_fn(self, grad: Tensor) -> Tensor:
        xp = select_backend(grad.array, self._mask)
        self._grad = xp.where(self._mask, grad.array, 0)

        result = to_tensor(self._grad)
        result.is_vector = grad.is_vector
        return result

    def forward(self, tensor: Tensor, mask: Tensor) -> Tensor:
        """
        Apply a binary mask to a numpy array, setting values to 0 where
        the mask is True
        """
        xp = select_backend(tensor.array, mask.array)
        assert _shapes_match(tensor, mask)
        assert (
            not mask.requires_grad
        ), "Cannot compute gradient of a binary mask"

        self._out = xp.where(mask.array, tensor.array, 0)
        self._mask = mask.array

        result = to_tensor(self._out)

        result.args = (tensor,)
        result.back_fns = (self.back_fn,)
        result.name = "bmask"
        result.is_vector = tensor.is_vector
        return result
