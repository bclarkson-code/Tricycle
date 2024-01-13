from tricycle_v2.tensor import Op, Tensor
from functools import wraps
from tricycle_v2.ops import to_tensor as to_tensor_op


def to_tensor(fn: Op) -> Op:
    """
    A decorator to convert non-tensor arguments to tensors
    """

    @wraps(fn)
    def wrapped(*args, **kwargs):
        args = [arg if isinstance(arg, Tensor) else to_tensor_op(arg) for arg in args]
        return fn(*args, **kwargs)

    return wrapped
