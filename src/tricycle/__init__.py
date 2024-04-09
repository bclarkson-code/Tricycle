from warnings import warn

CUPY_ENABLED = True
# try:
#     import cupy
#
#     cupy.array([1, 2, 3])
#     CUPY_ENABLED = True
# except ImportError:
#     warn("Could not find CuPY, disabling GPU features")
#     CUPY_ENABLED = False
# except Exception as e:
#     CUPY_ENABLED = False
#     warn(f"Failed to build cupy array: {e}. Disabling GPU features")
