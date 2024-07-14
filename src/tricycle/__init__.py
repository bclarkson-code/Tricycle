import threading
from warnings import warn

try:
    import cupy

    cupy.array([1, 2, 3]) * 2
    GPU_ENABLED = True
except ImportError:
    warn("Could not find CuPY, disabling GPU features")
    GPU_ENABLED = False
except Exception as e:
    GPU_ENABLED = False
    warn(f"Failed to build cupy array: {e}. Disabling GPU features")

# We use this to store state (e.g whether we're using 16 bit or not)
# we're using a threading.local to avoid problems with threads
TRICYCLE_CONTEXT = threading.local()

# you can modify this to use mixed precision all the time but the recommended
# method for mixed precision training is to use the
# tricycle/utils.py:UseMixedPrecision context manager
TRICYCLE_CONTEXT.use_mixed_precision = False
