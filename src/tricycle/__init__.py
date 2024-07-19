from warnings import warn

try:
    import cupy

    # check that we can create a cupy array and operate on it
    cupy.array([1, 2, 3]) * 2
    GPU_ENABLED = True
except ImportError:
    warn("Could not find CuPY, disabling GPU features")
    GPU_ENABLED = False
except Exception as e:
    GPU_ENABLED = False
    warn(f"Failed to build cupy array: {e}. Disabling GPU features")


class TricycleContext:
    # you can modify this to use mixed precision all the time but the
    # recommended method for mixed precision training is to use the
    # tricycle/utils.py:UseMixedPrecision context manager
    use_mixed_precision: bool = True

    # If we're using mixed precision, we'll want to scale the loss to help
    # with under and overflowing
    loss_scale_factor: int = 128


TRICYCLE_CONTEXT = TricycleContext()
