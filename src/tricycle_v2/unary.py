from string import ascii_letters, ascii_lowercase

import numpy as np

from tricycle_v2.ops import einsum, nothing, to_tensor
from tricycle_v2.tensor import Tensor

grad = False


def uadd(tensor, constant):
    """
    Add a constant, elementwise, to a tensor. The constant is not
    differentiable.
    """
    result = to_tensor(np.add(tensor, constant))
    result.args = (tensor,)
    result.back_fn = (nothing,)
    return result


def umul(tensor, constant):
    """
    Multiply a constant, elementwise, to a tensor. The constant is not
    differentiable.
    """
    constant_tensor = to_tensor(np.full_like(tensor, constant), requires_grad=False)
    indices = ascii_letters[: len(tensor.shape)]
    subscripts = f"{indices},{indices}->{indices}"
    return einsum(subscripts, tensor, constant_tensor)


def usub(arg_1, arg_2):
    """
    Subtract a constant, elementwise, from a tensor. The constant is not
    differentiable.
    """
    if isinstance(arg_1, Tensor) and np.isscalar(arg_2):
        return uadd(arg_1, -to_tensor(arg_2, requires_grad=False))
    elif isinstance(arg_2, Tensor) and np.isscalar(arg_1):
        return uadd(
            to_tensor(arg_1, requires_grad=False),
            umul(arg_2, -1),
        )
    else:
        raise NotImplementedError(
            f"Subtraction between {type(arg_1)} and {type(arg_2)}"
        )
