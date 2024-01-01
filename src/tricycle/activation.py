import numpy as np

from tricycle.ops import Tensor, einsum, tensor, to_tensor


@to_tensor
def relu(x: Tensor) -> Tensor:
    """
    Compute the relu of a tensor
    """
    result = tensor(np.maximum(x, 0))

    def diff_relu(arg: Tensor) -> Tensor:
        weight = (x > 0).astype(x.dtype)
        return einsum(weight, arg, subscripts="ij,ij->ij")

    result.back_fn = (diff_relu,)
    result.args = (x,)

    return result
