"""
Einsum is a generalisation of a large number of matrix operations.

You can use it by assigning each axis in your matrices a letter of the
alphabet (called an index). You can define the operation you want to perform
by simply listing the indices you want in your inputs and output, separated by
an arrow.

For example, you can define the transpose of a 2d tensor as follows:

>>> a = to_tensor([[1,2],[3,4]])
>>> Einsum("ij->ji")(a)
Tensor([[1. 3.]
 [2. 4.]], name=einsum ij->ji)

Here, we use einsum to swap indices i and j: a transpose.

There are only two rules to remember with einsum:
 - If an index does not appear in the output, any inputs that contain it
   will be summed along that axis:

    >>> Einsum("ij->i")(a)
    Tensor([3. 7.], name=einsum ij->i)

 - If an index appears in more than one input, the tensors will be multiplied
   along that axis

   >>> b = to_tensor([[5,6],[7,8])
   >>> Einsum("ij,jk->ik")(a,b)
    Tensor([[19. 22.]
     [43. 50.]], name=einsum ij,jk->ik)



You can use einsum to perform all of these operations:
"""

import itertools
import re
from typing import Sequence

from tricycle.tensor import Tensor, select_backend, to_tensor


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


class EinsumBackOp:
    """
    The backward operation for an einsum operation. This is done by
    swapping the indices and tensors for an input with the output.
    E.g "ij,jk->ik" with idx = 0 would become "ik,jk->ij"
    """

    def __init__(
        self, idx: int, tensors: Sequence[Tensor], subscript: Subscript
    ):
        self.idx = idx
        self.tensors = tensors
        self.subscript = subscript

        self.left_tensors, self.right_tensors = self._build_inputs()
        self.combined_subscript = self._build_subscript()

    def _build_inputs(self):
        left_tensors = self.tensors[: self.idx]

        # Special case for the last index
        if self.idx < len(self.tensors) - 1:
            right_tensors = self.tensors[self.idx + 1 :]
        else:
            right_tensors = []

        return left_tensors, right_tensors

    def _build_subscript(self):
        left_subscript = self.subscript.inputs[: self.idx]

        # Special case for the last index
        if self.idx < len(self.tensors) - 1:
            right_subscript = self.subscript.inputs[self.idx + 1 :]
        else:
            right_subscript = []

        combined_indices = [
            *left_subscript,
            self.subscript.output,
            *right_subscript,
        ]
        return Subscript.from_split(
            combined_indices, self.subscript.inputs[self.idx]
        )

    def __call__(self, tensor: Tensor):
        """
        Build the backward function for einsum. This is done by
        swapping the indices and tensors for an input with the output.
        E.g "ij,jk->ik" with idx = 0 would become "ik,jk->ij"
        """

        combined_tensors = [*self.left_tensors, tensor, *self.right_tensors]
        return Einsum(self.combined_subscript)(*combined_tensors)

    def __repr__(self):
        return f"EinsumBackOp({self.combined_subscript})"


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

        # To avoid adding a bunch of special cases for batched
        # operations, we replace any batched operations with
        # their non-batched counterparts
        subscript = Subscript(subscript.subscript.replace("z", ""))

        back_functions = []
        for idx in range(len(tensors)):
            back_op = EinsumBackOp(idx, tensors, subscript)
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
        einsum("ab,ab->")(x, xp.ones_like(x))

        # backward
        einsum(",ab->ab")(grad, xp.ones_like(x))
        """
        xp = select_backend(*tensors)
        if len(tensors) != 1:
            return subscript, tensors

        [tensor] = tensors
        ones = to_tensor(
            xp.ones(tensor.shape),
            is_batched=tensor.is_batched,
            requires_grad=False,
        )
        tensors = [tensor, ones]

        [index] = subscript.inputs
        output = subscript.output
        inputs = [index, index]
        subscript = Subscript.from_split(inputs, output)

        return subscript, tensors

    def _handle_batched(
        self, subscript: Subscript, tensors: Sequence[Tensor]
    ) -> tuple[Subscript, Sequence[Tensor], bool]:
        """
        If a tensor is labelled as being batched, add an extra dimension
        to its indices.
        """
        inputs = []
        batch_output = False
        for idx, tensor in zip(subscript.inputs, tensors):
            if tensor.is_batched:
                inputs.append(["z"] + idx)
                batch_output = True
            else:
                inputs.append(idx)
        output = subscript.output
        if batch_output:
            if "z" in subscript.subscript:
                raise ValueError(
                    "`z` cannot be used in an einsum subscript on "
                    "non-batched tensors because "
                    "it is reserved for batched indices."
                )
            output = ["z"] + output

        subscript = Subscript.from_split(inputs, output)
        return subscript, tensors, batch_output

    def _replace_infinity(self, tensors: Sequence[Tensor]):
        """
        If tensors contain infinity, temporarily replace them with the max
        value for that datatype
        """
        xp = select_backend(*tensors)
        processed = []
        for tensor in tensors:
            if not xp.isinf(tensor.array).any():
                processed.append(tensor)
                continue

            new_tensor = to_tensor(
                xp.nan_to_num(tensor.array),
                is_batched=tensor.is_batched,
            )
            new_tensor.args = tensor.args
            new_tensor.back_fns = tensor.back_fns
            new_tensor.name = tensor.name
            processed.append(new_tensor)

        return processed

    def __call__(self, *tensors: Tensor):
        xp = select_backend(*tensors)
        subscript, tensors, batch_output = self._handle_batched(
            self.subscript, tensors
        )
        subscript, tensors = self._handle_single_tensor(subscript, tensors)
        tensor_data = [t.array for t in tensors]
        result = to_tensor(xp.einsum(str(subscript), *tensor_data))
        if batch_output:
            result.is_batched = True

        result.args = tuple(tensors)
        result.back_fns = tuple(self._build_back_ops(tensors, subscript))
        result.name = f"einsum {self.subscript}"
        return result
