import numpy as np

from tricycle_v2.ops import to_tensor


def init_xavier(shape, name: str = ""):
    """
    Initialize a tensor with xavier/glorot initialisation
    """
    f_in, f_out = shape
    bound = np.sqrt(6) / np.sqrt(f_in + f_out)
    return to_tensor(np.random.uniform(low=-bound, high=bound, size=shape), name=name)
