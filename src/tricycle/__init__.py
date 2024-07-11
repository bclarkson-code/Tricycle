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
