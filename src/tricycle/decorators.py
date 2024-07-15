from functools import wraps

from tricycle.tensor import Op, Tensor


def to_tensor(fn: Op) -> Op:
    """
    A decorator to convert non-tensor arguments to tensors
    """

    @wraps(fn)
    def wrapped(*args, **kwargs):
        args = [
            arg if isinstance(arg, Tensor) else Tensor(arg) for arg in args
        ]
        return fn(*args, **kwargs)

    return wrapped
