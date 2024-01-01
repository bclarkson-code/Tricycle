import numpy as np

from tricycle.ops import tensor


def init_normal(shape, name: str = "", loc=0.0, scale=0.01):
    """
    Initialize a tensor with values sampled from a normal distribution
    """
    return tensor(np.random.normal(size=shape, loc=loc, scale=scale), name=name)


def init_zero(shape, name: str = ""):
    """
    Initialize a tensor with zeros
    """
    return tensor(np.zeros(shape), name=name)


def init_xavier(shape, name: str = ""):
    """
    Initialize a tensor with xavier/glorot initialisation
    """
    f_in, f_out = shape
    bound = np.sqrt(6) / np.sqrt(f_in + f_out)
    return tensor(np.random.uniform(low=-bound, high=bound, size=shape), name=name)
