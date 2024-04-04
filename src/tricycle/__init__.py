
from warnings import warn


try:
    import cupy
    CUPY_ENABLED = True
except ImportError:
    warn('Could not find CuPY, disabling GPU features')
    CUPY_ENABLED = False

