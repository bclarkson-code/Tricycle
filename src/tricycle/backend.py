"""
This file contains all of the numpy API that is used by tricycle
each function is decorated with a function that handles switching between
Cupy and Numpy (so we can switch between GPU and GPU)
"""

import cupy as cp
import numpy as np


def to_gpu(tensor):
    """
    Move a tensor to the GPU
    """
    return cp.asarray(tensor)


def from_cpu(tensor):
    """
    Move a tensor from the GPU
    """
    return cp.asnumpy(tensor)
