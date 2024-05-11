from abc import abstractmethod
from copy import copy
from typing import Sequence

from numpy.typing import ArrayLike

from tricycle.einsum import Einsum, Subscript
from tricycle.tensor import Tensor, to_tensor


class Op:
    """
    Base class for operations
    """

    _out: ArrayLike | None = None
    _grad: ArrayLike | None = None

    def __call__(self, tensor: Tensor, *args, **kwargs) -> Tensor:
        return self.forward(tensor, *args, **kwargs)

    @abstractmethod
    def forward(self, tensor: Tensor, *args, **kwargs) -> Tensor:
        raise NotImplementedError()


class Repeat(Op):
    def forward(self, tensor: Tensor, repeats: int):
        """
        Repeat a tensor along its final axis
        This is done my multiplying with a ones tensor the same shape as the
        desired output
        """
        xp = tensor.xp
        subscript = Subscript("...,...a->...a")
        new_shape = tensor.shape + (repeats,)
        ones = to_tensor(
            xp.ones(new_shape), is_vector=tensor.is_vector, requires_grad=False
        )

        return Einsum(subscript)(tensor, ones)


class Split(Op):
    _indices: tuple[int]
    _axis: int
    _n_splits: int
    _grad: list[ArrayLike]

    def back_fn(self, grad: Tensor, idx: int) -> Tensor:
        """
        The backwards operation for a split operation.
        Produces a tensor of zeros the same shape as the input
        except in the section that was split

        e.g
        >>> result = split([1,2,3,4], 2)
        >>> result
        [tensor([1, 2]), tensor([3, 4])]
        # set an arbitrary derivative for first split
        >>> result[0].grad = to_tensor([1,1])
        >>> undo_split(result[0].grad)
        [1, 1, 0, 0]
        """
        xp = grad.xp
        self._grad[idx] = xp.zeros(self._in_shape)

        indices = []
        for i in range(self._grad[idx].ndim):
            if i == self._axis % self._grad[idx].ndim:
                step = self._in_shape[i] // self._n_splits
                start = step * idx
                end = step * (idx + 1)
                indices.append(slice(start, end))
            else:
                indices.append(slice(None))
        self._grad[idx][tuple(indices)] = grad._data

        result = to_tensor(self._grad[idx])
        result.is_vector = grad.is_vector
        return result

    def forward(
        self, tensor: Tensor, n_splits: int, axis: int = -1
    ) -> Sequence[Tensor]:
        """
        Split a tensor along an axis into n_splits partitions
        """
        xp = tensor.xp

        assert isinstance(n_splits, int)

        self._out = xp.split(tensor._data, n_splits, axis=axis)
        self._in_shape = tensor.shape
        self._axis = axis
        self._n_splits = n_splits
        self._grad = [None] * n_splits

        results = []
        for idx, result in enumerate(self._out):
            # the back_fn depends on index so we need to
            # dynamically create this function
            def back_fn(grad, idx=idx):
                return self.back_fn(grad, idx=idx)

            result = to_tensor(result)
            result.back_fns = (back_fn,)
            result.args = (tensor,)
            result.is_vector = tensor.is_vector
            results.append(result)
        return results


class Reshape(Op):
    _original_shape: Sequence[int]

    def back_fn(self, grad: Tensor) -> Tensor:  # sourcery skip: assign-if-exp
        xp = grad.xp

        self._grad = xp.reshape(grad._data, self._original_shape)
        result = to_tensor(self._grad)
        result.is_vector = grad.is_vector
        return result

    def forward(self, tensor: Tensor, shape: Sequence[int]) -> Tensor:
        xp = tensor.xp
        if tensor.is_vector:
            shape = [tensor.shape[0]] + list(shape)

        self._out = xp.reshape(tensor._data, shape)
        self._original_shape = tensor.shape

        result = to_tensor(self._out)
        result.args = (tensor,)
        result.back_fns = (self.back_fn,)
        result.is_vector = tensor.is_vector

        return result


class Mean(Op):
    def forward(self, tensor: Tensor) -> Tensor:
        """
        Find the mean of a tensor along the final axis
        """
        if tensor.ndim == 1 and tensor.shape[0] == 1:
            return tensor

        return tensor.e("...a->...") / tensor.shape[-1]
