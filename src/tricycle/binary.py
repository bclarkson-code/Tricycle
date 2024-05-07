from functools import partial

from numpy.typing import ArrayLike

from tricycle.ops import Einsum, Op
from tricycle.tensor import Tensor, nothing, select_backend, to_tensor
from tricycle.unary import UDiv, UMul


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
            f"Shapes {shape_1} and {shape_2} do not match: {tensor_1._data.shape}, {tensor_2._data.shape}"
        )
    return shape_1 == shape_2


class BAdd(Op):
    def __call__(self, tensor_1: Tensor, tensor_2: Tensor) -> Tensor:
        """
        Applies the cosine function, elementwise, to a tensor
        """
        xp = select_backend(tensor_1._data, tensor_2._data)

        assert _shapes_match(tensor_1, tensor_2)
        self._out = xp.add(tensor_1._data, tensor_2._data)

        result = to_tensor(self._out)
        result.args = (tensor_1, tensor_2)
        result.back_fns = (nothing, nothing)
        result.name = "badd"

        if tensor_1.is_vector or tensor_2.is_vector:
            result.is_vector = True

        return result


class BSub(Op):
    def back_fn_2(self, grad: Tensor) -> Tensor:
        self._grad = -grad._data
        result = to_tensor(self._grad)
        result.is_vector = grad.is_vector
        return result

    def __call__(self, tensor_1: Tensor, tensor_2: Tensor) -> Tensor:
        """
        Subtract one tensor from another
        """
        xp = select_backend(tensor_1._data, tensor_2._data)

        assert _shapes_match(tensor_1, tensor_2)
        self._out = xp.subtract(tensor_1._data, tensor_2._data)

        result = to_tensor(self._out)
        result.args = (tensor_1, tensor_2)
        result.back_fns = (nothing, self.back_fn_2)
        result.name = "badd"

        if tensor_1.is_vector or tensor_2.is_vector:
            result.is_vector = True

        return result


class BMul(Op):
    def __call__(self, tensor_1: Tensor, tensor_2: Tensor) -> Tensor:
        """
        Multiply the elements of two tensors together, elementwise

        The two tensors must have the same shape
        """
        assert _shapes_match(tensor_1, tensor_2)

        result = Einsum("...,...->...")(tensor_1, tensor_2)
        result.name = "bmul"
        return result


class BDiv(Op):
    def __call__(self, tensor_1: Tensor, tensor_2: Tensor) -> Tensor:
        """
        Divide the elements of two tensors together, elementwise

        The two tensors must have the same shape
        """
        mul = BMul()
        div = UDiv()

        return mul(tensor_1, div(1, tensor_2))


class BMax(Op):
    _is_bigger_1: ArrayLike | None
    _is_bigger_2: ArrayLike | None

    def back_fn_1(self, grad: Tensor) -> Tensor:
        self._grad_1 = grad._data * self._is_bigger_1
        result = to_tensor(self._grad_1)
        result.is_vector = grad.is_vector
        return result

    def back_fn_2(self, grad: Tensor) -> Tensor:
        self._grad_2 = grad._data * self._is_bigger_2
        result = to_tensor(self._grad_2)
        result.is_vector = grad.is_vector
        return result

    def __call__(self, tensor_1: Tensor, tensor_2: Tensor) -> Tensor:
        """
        Compare two tensors elementwise, returning the maximum
        of each pair of elements

        The two tensors must have the same shape
        if elements are equal, return the first
        """
        xp = select_backend(tensor_1._data, tensor_2._data)
        assert _shapes_match(tensor_1, tensor_2)

        self._out = xp.maximum(tensor_1._data, tensor_2._data)

        self._is_bigger_1 = tensor_1._data > tensor_2._data
        self._is_bigger_2 = tensor_1._data <= tensor_2._data

        result = to_tensor(self._out)
        result.args = (tensor_1, tensor_2)
        result.back_fns = (self.back_fn_1, self.back_fn_2)
        result.name = "bmax"
        result.is_vector = tensor_1.is_vector or tensor_2.is_vector
        return result


class BMin(Op):
    _is_smaller_1: Tensor | None
    _is_smaller_2: Tensor | None

    def back_fn_1(self, grad: Tensor) -> Tensor:
        self._grad_1 = grad._data * self._is_smaller_1
        result = to_tensor(self._grad_1)
        result.is_vector = grad.is_vector
        return result

    def back_fn_2(self, grad: Tensor) -> Tensor:
        self._grad_2 = grad._data * self._is_smaller_2
        result = to_tensor(self._grad_2)
        result.is_vector = grad.is_vector
        return result

    def __call__(self, tensor_1: Tensor, tensor_2: Tensor) -> Tensor:
        """
        Compare two tensors elementwise, returning the maximum
        of each pair of elements

        The two tensors must have the same shape
        if elements are equal, return the first
        """
        xp = select_backend(tensor_1._data, tensor_2._data)
        assert _shapes_match(tensor_1, tensor_2)

        self._out = xp.minimum(tensor_1._data, tensor_2._data)

        self._is_smaller_1 = tensor_1._data < tensor_2._data
        self._is_smaller_2 = tensor_1._data >= tensor_2._data

        result = to_tensor(self._out)
        result.args = (tensor_1, tensor_2)
        result.back_fns = (self.back_fn_1, self.back_fn_2)
        result.name = "bmax"
        result.is_vector = tensor_1.is_vector or tensor_2.is_vector
        return result


class BMask(Op):
    _mask: ArrayLike | None = None

    def back_fn(self, grad: Tensor) -> Tensor:
        xp = select_backend(grad._data, self._mask)
        self._grad = xp.where(self._mask, grad._data, 0)

        result = to_tensor(self._grad)
        result.is_vector = grad.is_vector
        return result

    def __call__(self, tensor: Tensor, mask: Tensor) -> Tensor:
        """
        Apply a binary mask to a numpy array, setting values to 0 where
        the mask is True
        """
        xp = select_backend(tensor._data, mask._data)
        assert _shapes_match(tensor, mask)
        assert (
            not mask.requires_grad
        ), "Cannot compute gradient of a binary mask"

        self._out = xp.where(mask._data, tensor._data, 0)
        self._mask = mask._data

        result = to_tensor(self._out)

        result.args = (tensor,)
        result.back_fns = (self.back_fn,)
        result.name = "bmask"
        result.is_vector = tensor.is_vector
        return result
