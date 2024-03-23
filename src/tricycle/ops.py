import numpy as np

from tricycle.einsum import Einsum, Subscript
from tricycle.reduce import rmax
from tricycle.tensor import Tensor, to_tensor


def repeat(subscript: Subscript | str, tensor: Tensor, repeats: int):
    """
    Repeat a tensor along some indices, according to the subscript.
    Note: This is mathematically equivalent to Einsumming the tensor
    with a one tensor
    """
    if isinstance(subscript, str):
        subscript = Subscript(subscript)

    unique_indices = set(",".join(subscript.inputs))

    unset_indices = "".join(set(subscript.output) - unique_indices)
    one_shape = [repeats] * len(unset_indices)

    ones = to_tensor(np.ones(one_shape), requires_grad=False)
    inputs = [unset_indices] + subscript.inputs
    new_subscript = Subscript.from_split(inputs, subscript.output)
    return Einsum(new_subscript)(ones, tensor)


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
    from tricycle.unary import uexp

    # normalise
    largest_element = rmax(tensor, "a->").repeat("->a", tensor.shape[-1])
    tensor = tensor - largest_element

    numerator = uexp(tensor)
    denominator = numerator.e("a->").repeat("->a", tensor.shape[-1])
    return bdiv(numerator, denominator)


def arange(*args, **kwargs):
    return to_tensor(np.arange(*args, **kwargs))
