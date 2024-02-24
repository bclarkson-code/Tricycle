from functools import partial
from string import ascii_lowercase
from typing import Sequence

import numpy as np

from tricycle.tensor import Tensor, to_tensor


class AlreadyVectorised(Exception):
    pass


class einsum:
    def __init__(self, subscript: str):
        self.subscript = subscript
        self.indices, self.output = _parse_subscripts(subscript)

    def _generate_back_fns(self, tensors: Sequence[Tensor]):
        assert len(tensors) == len(self.indices)
        back_functions = []
        for idx in range(len(tensors)):

            def back_fn(tensor: Tensor, idx: int = idx):
                left_tensors = tensors[:idx]
                left_subscript = self.indices[:idx]

                right_tensors = tensors[idx + 1 :] if idx < len(tensors) - 1 else []
                right_subscript = (
                    self.indices[idx + 1 :] if idx < len(tensors) - 1 else []
                )

                subscript = left_subscript + [self.output] + right_subscript
                subscript = ",".join(subscript) + "->" + self.indices[idx]

                fn_args = [*left_tensors, tensor, *right_tensors]
                return einsum(subscript)(*fn_args)

            back_functions.append(back_fn)

        return back_functions

    def __call__(self, *tensors: Tensor):
        result = to_tensor(np.einsum(self.subscript, *tensors))
        result.args = tuple(tensors)
        result.back_fns = tuple(self._generate_back_fns(tensors))
        result.name = f"einsum {self.subscript}"
        return result


def repeat(subscripts, tensor, out_shape):
    """
    Repeat a tensor along some indices, according to the subscript.
    Note: This is mathematically equivalent to einsumming the tensor
    with a one tensor
    """
    indices, output = _parse_subscripts(subscripts)
    assert len(indices) == 1
    [index] = indices

    assert len(output) == len(out_shape), "Output shape does not match subscripts"

    one_indices = ""
    one_shape = []
    for size, out_idx in zip(out_shape, output):
        if out_idx not in index:
            one_indices += out_idx
            one_shape.append(size)

    ones = to_tensor(np.ones(one_shape), requires_grad=False)
    new_subscript = f"{one_indices},{index}->{output}"
    return einsum(new_subscript)(ones, tensor)


def _parse_subscripts(subscripts: str) -> tuple[list[str], str]:
    """
    Parse a subscripts string into a list of indices and a result
    """
    indices, result = subscripts.split("->")
    indices = indices.split(",")
    return indices, result


def nothing(tensor):
    """
    Return a tensor
    """
    return tensor


def softmax(tensor):
    """
    Apply softmax. The softmax is only applied to the final
    dimension of the tensor
    Note: the tensor is normalised for numeric stability
    """
    from tricycle.binary import bdiv
    from tricycle.reduce import radd
    from tricycle.unary import uexp

    indices = ascii_lowercase[: len(tensor.shape)]
    reduce_subscript = f"{indices}->{indices[:-1]}"

    expand_subscript = f"{indices[:-1]}->{indices}"
    normalised = tensor
    exponentiated = uexp(normalised)

    denom = radd(exponentiated, reduce_subscript)
    denom = repeat(expand_subscript, denom, tensor.shape)
    return bdiv(exponentiated, denom)
