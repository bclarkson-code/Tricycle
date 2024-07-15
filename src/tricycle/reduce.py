from tricycle.einsum import Einsum, Subscript
from tricycle.ops import Op
from tricycle.tensor import Tensor


class ReduceMax(Op):
    def __call__(self, tensor: Tensor, subscript: Subscript | str):
        """
        Generate an indicator tensor that, when einsummed with the tensor,
        results in a tensor that is equal to the result of max applied along
        the indices that dont appear in the output of the subscript
        """
        if isinstance(subscript, str):
            subscript = Subscript(subscript)

        assert (
            len(subscript.inputs) == 1
        ), f"Can only reduce a single tensor at a time. Indices suggeststed: {len(subscript.inputs)} tensors: {subscript.inputs}"

        [idx] = subscript.inputs

        reduce_along_axes = [
            i for i, char in enumerate(idx) if char not in subscript.output
        ]

        if not reduce_along_axes:
            return tensor

        indicator = tensor.array == tensor.xp.max(
            tensor.array, axis=tuple(reduce_along_axes), keepdims=True
        )
        indicator = Tensor(
            indicator, requires_grad=False, is_batched=tensor.is_batched
        )
        indicator.array = indicator.array.astype(tensor.xp.int8)

        new_subscript = Subscript.from_split([idx, idx], subscript.output)

        result = Einsum(new_subscript)(tensor, indicator)
        result.name = f"min({new_subscript})"

        return result


class ReduceMin(Op):
    def __call__(self, tensor: Tensor, subscript: Subscript | str):
        """
        Generate an indicator tensor that, when einsummed with the tensor, results
        in a tensor that is equal to the result of min applied along the indices
        that dont appear in the output of the subscript
        """
        if isinstance(subscript, str):
            subscript = Subscript(subscript)

        assert (
            len(subscript.inputs) == 1
        ), f"Can only reduce a single tensor at a time. Indices suggeststed: {len(subscript.inputs)} tensors: {subscript.inputs}"

        [idx] = subscript.inputs

        reduce_along_axes = [
            i for i, char in enumerate(idx) if char not in subscript.output
        ]

        if not reduce_along_axes:
            return tensor

        indicator = tensor.array == tensor.xp.min(
            tensor.array, axis=tuple(reduce_along_axes), keepdims=True
        )
        indicator = Tensor(
            indicator, requires_grad=False, is_batched=tensor.is_batched
        )
        indicator.array = indicator.array.astype(tensor.xp.int8)

        new_subscript = Subscript.from_split([idx, idx], subscript.output)

        result = Einsum(new_subscript)(tensor, indicator)
        result.name = f"min({new_subscript})"

        return result
