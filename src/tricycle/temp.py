from pathlib import Path

import cupy as cp

from tricycle.layers import Layer
from tricycle.tensor import Tensor


class GeLUV2(Layer):
    """
    llm.c implementation of GeLU
    """

    def __init__(self):
        llm_c_binary = Path(__file__).parent.parent / "llm.c/kernels.ptx"
        self.module = cp.RawModule(path=str(llm_c_binary))
        self.forward_kernel = self.module.get_function("gelu_forward_kernel2")

    def forward(self, tensor: Tensor):
        GRID_SIZE = 32
        BLOCK_SIZE = 32
        out = cp.empty_like(tensor.array)
        self.forward_kernel(GRID_SIZE, BLOCK_SIZE, out, tensor.array)
        breakpoint()
        return Tensor(out)


if __name__ == "__main__":

    # Read all the necessary CUDA files
    cuda_common_path = (
        Path(__file__).parent.parent / "llm.c/llmc/cuda_common.h"
    )
    cuda_utils_path = (
        Path(__file__).parent.parent / "llm.c/llmc/cuda_utils.cuh"
    )
    gelu_path = Path(__file__).parent.parent / "llm.c/llmc/gelu.cuh"
    kernels_path = Path(__file__).parent.parent / "llm.c/kernels.cu"

    with open(cuda_common_path, "r") as f:
        cuda_common_content = f.read()

    with open(cuda_utils_path, "r") as f:
        cuda_utils_content = f.read()

    with open(gelu_path, "r") as f:
        gelu_content = f.read()

    with open(kernels_path, "r") as f:
        kernels_content = f.read()

    # Combine all CUDA code into a single string
    cuda_code = f"""
    {cuda_common_content}
    {cuda_utils_content}
    {gelu_content}
    {kernels_content}
    """

    # Set up include paths
    cuda_path = "/home/ben/mambaforge/envs/tricycle/"  # Replace this with your actual CUDA path
    include_paths = [
        f"{cuda_path}/include",
        f"{cuda_path}/lib",
        "/usr/include",
        f"{cuda_path}/targets/x86_64-linux/include",
        str(
            Path(__file__).parent.parent / "llm.c"
        ),  # path to your project's include directory
    ]

    # Compile the CUDA code
    cuda_options = (
        # "--gpu-architecture=compute_70",  # Adjust this to match your GPU architecture
        "--std=c++17",
        # "-DENABLE_BF16",
    ) + tuple(f"-I{path}" for path in include_paths)

    module = cp.RawModule(
        code=cuda_code,
        options=cuda_options,
        name_expressions=["gelu_forward_kernel3"],
    )

    # # Try to get the kernel function
    # kernel_names = module.get_function_names()
    # print("Available kernels:", kernel_names)

    forward_kernel = module.get_function("gelu_forward_kernel3")
