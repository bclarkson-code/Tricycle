import numpy as np

from tricycle.ops import Tensor
from tricycle.tensor import DEFAULT_DTYPE


def init_xavier(shape: tuple[int, int], name: str = ""):
    """
    Initialize a tensor with xavier/glorot initialisation
    """
    f_in, f_out = shape
    bound = np.sqrt(6) / np.sqrt(f_in + f_out)
    out = Tensor(
        np.random.uniform(low=-bound, high=bound, size=shape),
        dtype=DEFAULT_DTYPE,
        name=name,
    )
    return out
