from warnings import warn

try:
    import cupy

    cupy.array([1, 2, 3])
    CUPY_ENABLED = True
except ImportError:
    warn("Could not find CuPY, disabling GPU features")
    CUPY_ENABLED = False
except Exception:
    warn("Failed to build cupy array. Disabling GPU features")
