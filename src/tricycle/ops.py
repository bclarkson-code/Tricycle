from abc import abstractmethod
from typing import Sequence

from numpy.typing import ArrayLike

from tricycle.einsum import Einsum, Subscript
from tricycle.tensor import Tensor, to_tensor


class Op:
    """
    Base class for operations
    """

    _out: ArrayLike | None = None

    def __call__(self, *args, **kwargs) -> Tensor:
        return self.forward(*args, **kwargs)

    @abstractmethod
    def forward(self, *args, **kwargs) -> Tensor:
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
            xp.ones(new_shape),
            is_batched=tensor.is_batched,
            requires_grad=False,
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

        # TODO: this loop is really slow and should be replaced
        indices = []
        for i in range(self._grad[idx].ndim):
            if i == self._axis % self._grad[idx].ndim:
                step = self._in_shape[i] // self._n_splits
                start = step * idx
                end = step * (idx + 1)
                indices.append(slice(start, end))
            else:
                indices.append(slice(None))
        self._grad[idx][tuple(indices)] = grad.array

        result = to_tensor(self._grad[idx])
        result.is_batched = grad.is_batched
        return result

    def forward(
        self, tensor: Tensor, n_splits: int, axis: int = -1
    ) -> Sequence[Tensor]:
        """
        Split a tensor along an axis into n_splits partitions
        """
        xp = tensor.xp

        assert isinstance(n_splits, int)

        self._out = xp.split(tensor.array, n_splits, axis=axis)
        self._in_shape = tensor.shape
        self._axis = axis
        self._n_splits = n_splits
        self._grad = [None] * n_splits

        # TODO: this loop is really slow and should be replaced
        results = []
        for idx, result in enumerate(self._out):
            # the back_fn depends on index so we need to
            # dynamically create this function
            def back_fn(grad, idx=idx):
                return self.back_fn(grad, idx=idx)

            result = to_tensor(result)
            result.back_fns = (back_fn,)
            result.args = (tensor,)
            result.is_batched = tensor.is_batched
            results.append(result)
        return results


class Reshape(Op):
    """
    Reshape a tensor. The new shape needs to have the same number of elements
    as the original, but can have any number of dimensions
    """

    _original_shape: Sequence[int]

    def back_fn(self, grad: Tensor) -> Tensor:
        xp = grad.xp

        self._grad = xp.reshape(grad.array, self._original_shape)

        return Tensor(array=self._grad, is_batched=grad.is_batched)

    def forward(self, tensor: Tensor, shape: Sequence[int]) -> Tensor:
        """
        Reshape a tensor. The new shape needs to have the same number of elements
        as the original, but can have any number of dimensions
        """
        xp = tensor.xp

        # if the tensor is batched, don't include the first dimension in
        # the reshape
        if tensor.is_batched:
            shape = [tensor.shape[0]] + list(shape)

        self._out = xp.reshape(tensor.array, shape)
        self._original_shape = tensor.shape

        return Tensor(
            array=self._out,
            args=(tensor,),
            back_fns=(self.back_fn,),
            name="reshape",
            is_batched=tensor.is_batched,
        )


class Mean(Op):
    """
    Find the mean of a tensor along the final axis
    """

    def forward(self, tensor: Tensor) -> Tensor:
        """
        Find the mean of a tensor along the final axis
        """
        if tensor.ndim == 1 and tensor.shape[0] == 1:
            return tensor

        return tensor.e("...a->...") / tensor.shape[-1]
