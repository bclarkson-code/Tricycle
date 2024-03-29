import itertools
import re
from typing import Sequence

import numpy as np

from tricycle.tensor import Tensor, to_tensor


class Subscript:
    """
    A string that defines an einsum operation
    """

    subscript: str
    inputs: list[list[str]]
    output: list[str]
    _index_pattern: re.Pattern = re.compile(r"(?:[A-Za-z]|(?:\.{3}))")

    def __init__(self, subscript: str):
        self.subscript = subscript
        self.inputs, self.output = self.split()

    def split(self) -> tuple[list[list[str]], list[str]]:
        """
        Parse a subscripts string into a list of indices and a result
        """
        indices, result = self.subscript.split("->")
        indices = indices.split(",")
        indices = [re.findall(self._index_pattern, idx) for idx in indices]
        result = re.findall(self._index_pattern, result)
        return indices, result

    @staticmethod
    def join(indices: list[list[str]], result: list[str]) -> str:
        inputs_string = ",".join(["".join(idx) for idx in indices])
        outputs_string = "".join(result)
        return f"{inputs_string}->{outputs_string}"

    @classmethod
    def from_split(cls, indices: list[list[str]], result: list[str]):
        return cls(cls.join(indices, result))

    @property
    def unique_input_indices(self) -> set[str]:
        all_inputs = itertools.chain(*self.inputs)
        return set(all_inputs)

    def __repr__(self):
        return self.subscript

    def __str__(self):
        return self.subscript


class Einsum:
    subscript: Subscript

    def __init__(self, subscript: str | Subscript):
        if isinstance(subscript, str):
            subscript = Subscript(subscript)
        self.subscript = subscript

    def _build_back_ops(self, tensors: Sequence[Tensor], subscript: Subscript):
        """
        Figure out the backward operation for each input
        """
        assert len(tensors) == len(subscript.inputs)

        # To avoid adding a bunch of special cases for vectorised
        # operations, we replace any vectorised operations with
        # their non-vectorised counterparts
        subscript = Subscript(subscript.subscript.replace("z", ""))

        back_functions = []
        for idx in range(len(tensors)):

            def back_op(tensor: Tensor, idx: int = idx):
                """
                Build the backward function for einsum. This is done by
                swapping the indices and tensors for an input with the output.
                E.g "ij,jk->ik" with idx = 0 would become "ik,jk->ij"
                """
                left_tensors = tensors[:idx]
                left_subscript = subscript.inputs[:idx]

                # Special case for the last index
                if idx < len(tensors) - 1:
                    right_tensors = tensors[idx + 1 :]
                    right_subscript = subscript.inputs[idx + 1 :]
                else:
                    right_tensors = []
                    right_subscript = []

                combined_tensors = [*left_tensors, tensor, *right_tensors]
                combined_indices = [
                    *left_subscript,
                    subscript.output,
                    *right_subscript,
                ]

                combined_subscript = Subscript.from_split(
                    combined_indices, subscript.inputs[idx]
                )
                return Einsum(combined_subscript)(*combined_tensors)

            back_functions.append(back_op)

        return back_functions

    def _handle_single_tensor(
        self, subscript: Subscript, tensors: Sequence[Tensor]
    ) -> tuple[Subscript, Sequence[Tensor]]:
        """
        If there is only one tensor, we need to insert a matrix of ones
        to allow for expansion operations

        E.g the reverse operation for einsum("ab->")(x) would be
        einsum("->ab")(grad), which is not defined. However, if we inject
        some ones that are the same shape as the input, the forward operation
        is unaltered while the backward operation is now fully defined:

        # forward
        einsum("ab,ab->")(x, np.ones_like(x))

        # backward
        einsum(",ab->ab")(grad, np.ones_like(x))
        """
        if len(tensors) != 1:
            return subscript, tensors

        [tensor] = tensors
        ones = to_tensor(
            np.ones_like(tensor),
            is_vector=tensor.is_vector,
            requires_grad=False,
        )
        tensors = [tensor, ones]

        [index] = subscript.inputs
        output = subscript.output
        inputs = [index, index]
        subscript = Subscript.from_split(inputs, output)

        return subscript, tensors

    def _handle_vectorised(
        self, subscript: Subscript, tensors: Sequence[Tensor]
    ) -> tuple[Subscript, Sequence[Tensor], bool]:
        """
        If a tensor is labelled as being vectorised, add an extra dimension
        to its indices.
        """
        inputs = []
        vectorise_output = False
        for idx, tensor in zip(subscript.inputs, tensors):
            if tensor.is_vector:
                inputs.append(["z"] + idx)
                vectorise_output = True
            else:
                inputs.append(idx)
        output = subscript.output
        if vectorise_output:
            if "z" in subscript.subscript:
                raise ValueError(
                    "`z` cannot be used in an einsum subscript on "
                    "non-vectorised tensors because "
                    "it is reserved for vectorised indices."
                )
            output = ["z"] + output

        subscript = Subscript.from_split(inputs, output)
        return subscript, tensors, vectorise_output

    def _replace_infinity(self, tensors: Sequence[Tensor]):
        """
        If tensors contain infinity, temporarily replace them with the max
        value for that datatype
        """
        processed = []
        for tensor in tensors:
            if not np.isinf(tensor.data).any():
                processed.append(tensor)
                continue

            new_tensor = to_tensor(
                np.nan_to_num(tensor.data), is_vector=tensor.is_vector
            )
            new_tensor.args = tensor.args
            new_tensor.back_fn = tensor.back_fn
            new_tensor.name = tensor.name
            processed.append(new_tensor)

        return processed

    def __call__(self, *tensors: Tensor, replace_inf=False):
        if replace_inf:
            tensors = self._replace_infinity(tensors)
        subscript, tensors, vectorise_output = self._handle_vectorised(
            self.subscript, tensors
        )
        subscript, tensors = self._handle_single_tensor(subscript, tensors)
        result = to_tensor(np.einsum(str(subscript), *tensors))
        if vectorise_output:
            result.is_vector = True

        result.args = tuple(tensors)
        result.back_fn = tuple(self._build_back_ops(tensors, subscript))
        result.name = f"einsum {self.subscript}"
        return result
