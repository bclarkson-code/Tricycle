import numbers

from numpy.typing import ArrayLike
from scipy.special import erf as np_erf

from tricycle import CUPY_ENABLED
from tricycle.ops import Op
from tricycle.tensor import Tensor, nothing, to_tensor

grad = False


class UAdd(Op):
    def __call__(self, tensor: Tensor, constant: float) -> Tensor:
        """
        Add a constant, elementwise, to a tensor. The constant is not
        differentiable.
        """
        xp = tensor.xp

        assert isinstance(tensor, Tensor)
        assert isinstance(constant, numbers.Number)

        self._out = xp.add(tensor._data, constant, dtype=tensor.dtype)

        result = to_tensor(self._out)
        result.args = (tensor,)
        result.back_fns = (nothing,)
        result.name = f"+ {constant}"
        result.is_vector = tensor.is_vector
        return result


class UMul(Op):
    _constant: float

    def back_fn(self, grad: Tensor) -> Tensor:
        xp = grad.xp

        self._grad = xp.multiply(grad._data, self._constant)
        result = to_tensor(self._grad)
        result.is_vector = grad.is_vector
        return result

    def __call__(self, tensor: Tensor, constant: float) -> Tensor:
        """
        Multiply a constant, elementwise, to a tensor. The constant is not
        differentiable.
        """
        xp = tensor.xp

        assert isinstance(tensor, Tensor)
        assert xp.isscalar(constant)

        self._out = xp.multiply(tensor._data, constant)
        self._constant = constant

        result = to_tensor(self._out)
        result.args = (tensor,)
        result.back_fns = (self.back_fn,)
        result.name = f"+ {constant}"
        result.is_vector = tensor.is_vector
        return result


class USub(Op):
    def __call__(self, tensor: Tensor, constant: float) -> Tensor:
        """
        Subtract a constant, elementwise, from a tensor. The constant is not
        differentiable.
        """
        return UAdd()(tensor, -constant)


class UPow(Op):
    input: ArrayLike
    constant: float

    def back_fn(self, grad: Tensor) -> Tensor:
        xp = grad.xp

        self._grad = xp.power(
            self.input._data, self.constant - 1, dtype=self.input.dtype
        )
        self._grad *= self.constant * grad._data

        result = to_tensor(self._grad)
        result.is_vector = grad.is_vector
        return result

    def __call__(self, tensor: Tensor, constant: float) -> Tensor:
        """
        Raise a tensor to a constant, elementwise. The constant is not
        differentiable.
        """
        xp = tensor.xp

        assert isinstance(tensor, Tensor)
        assert xp.isscalar(constant)

        self._out = xp.power(tensor._data, constant)
        self.input = tensor
        self.constant = constant

        result = to_tensor(self._out)
        result.args = (tensor,)
        result.back_fns = (self.back_fn,)
        result.name = f"^ {constant}"
        result.is_vector = tensor.is_vector

        return result


class UDiv(Op):
    # TODO: manually define the derivative instead of using other operations
    def __call__(self, constant: float, tensor: Tensor) -> Tensor:
        """
        Divide a constant by a tensor, elementwise. The constant is not
        differentiable.
        """
        upow = UPow()
        umul = UMul()
        return umul(upow(tensor, -1.0), constant)


class UMax(Op):
    is_bigger: Tensor

    def back_fn(self, grad: Tensor) -> Tensor:
        self._grad = grad._data * self.is_bigger._data

        result = to_tensor(self._grad)
        result.is_vector = grad.is_vector
        return result

    def __call__(self, tensor: Tensor, constant: float) -> Tensor:
        """
        Return the max of the tensor and the constant,
        elementwise. The constant is not differentiable.
        """
        xp = tensor.xp

        assert isinstance(tensor, Tensor)
        assert xp.isscalar(constant)

        self._out = xp.maximum(tensor._data, constant, dtype=tensor.dtype)

        self.is_bigger = tensor > constant
        self.is_bigger.is_vector = tensor.is_vector

        result = to_tensor(self._out)
        result.args = (tensor,)
        result.back_fns = (self.back_fn,)
        result.name = f"> {constant}"
        result.is_vector = tensor.is_vector

        return result


class UMin(Op):
    is_smaller: Tensor

    def back_fn(self, grad: Tensor) -> Tensor:
        self._grad = grad._data * self.is_smaller._data

        result = to_tensor(self._grad)
        result.is_vector = grad.is_vector
        return result

    def __call__(self, tensor: Tensor, constant: float) -> Tensor:
        """
        Return the max of the tensor and the constant,
        elementwise. The constant is not differentiable.
        """
        xp = tensor.xp

        assert isinstance(tensor, Tensor)
        assert xp.isscalar(constant)

        self._out = xp.minimum(tensor._data, constant, dtype=tensor.dtype)

        self.is_smaller = tensor < constant
        self.is_smaller.is_vector = tensor.is_vector

        result = to_tensor(self._out)
        result.args = (tensor,)
        result.back_fns = (self.back_fn,)
        result.name = f"> {constant}"
        result.is_vector = tensor.is_vector

        return result


class UExp(Op):
    def back_fn(self, grad: Tensor) -> Tensor:
        self._grad = grad._data * self._out

        result = to_tensor(self._grad)
        result.is_vector = grad.is_vector
        return result

    def __call__(self, tensor: Tensor) -> Tensor:
        """
        Raise every element of a tensor to the power of e
        """
        xp = tensor.xp

        self._out = xp.exp(tensor._data)

        result = to_tensor(self._out)
        result.args = (tensor,)
        result.back_fns = (self.back_fn,)
        result.name = "exp"
        result.is_vector = tensor.is_vector
        return result


class ULog(Op):
    REALLY_SMALL_NUMBER = 1e-8

    _input: ArrayLike | None = None

    def back_fn(self, grad: Tensor) -> Tensor:
        xp = grad.xp
        denominator = self._input + self.REALLY_SMALL_NUMBER
        self._grad = grad._data * xp.divide(1, denominator)

        result = to_tensor(self._grad)
        result.is_vector = grad.is_vector
        return result

    def __call__(self, tensor: Tensor) -> Tensor:
        """
        Raise every element of a tensor to the power of e
        """
        xp = tensor.xp

        self._out = xp.log(tensor._data)
        self._input = tensor._data

        result = to_tensor(self._out)

        result.args = (tensor,)
        result.back_fns = (self.back_fn,)
        result.name = "log"
        result.is_vector = tensor.is_vector
        return result


class USin(Op):
    _input: ArrayLike | None = None

    def back_fn(self, grad: Tensor) -> Tensor:
        xp = grad.xp

        self._grad = grad._data * xp.cos(self._input)

        result = to_tensor(self._grad)
        result.is_vector = grad.is_vector
        return result

    def __call__(self, tensor: Tensor) -> Tensor:
        """
        Applies the sine function, elementwise, to a tensor
        """
        xp = tensor.xp

        self._out = xp.sin(tensor._data)
        self._input = tensor._data

        result = to_tensor(self._out)
        result.args = (tensor,)
        result.back_fns = (self.back_fn,)
        result.name = "sin"
        result.is_vector = tensor.is_vector
        return result


class UCos(Op):
    _input: ArrayLike | None = None

    def back_fn(self, grad: Tensor) -> Tensor:
        xp = grad.xp

        self._grad = grad._data * -xp.sin(self._input)

        result = to_tensor(self._grad)
        result.is_vector = grad.is_vector
        return result

    def __call__(self, tensor: Tensor) -> Tensor:
        """
        Applies the cosine function, elementwise, to a tensor
        """
        xp = tensor.xp

        self._out = xp.cos(tensor._data)
        self._input = tensor._data

        result = to_tensor(self._out)
        result.args = (tensor,)
        result.back_fns = (self.back_fn,)
        result.name = "cos"
        result.is_vector = tensor.is_vector
        return result


class USqrt(Op):
    def __call__(self, tensor: Tensor):
        """
        Apply the square root function
        """
        upow = UPow()
        return upow(tensor, 0.5)
