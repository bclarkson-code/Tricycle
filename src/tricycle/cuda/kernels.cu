
#include "cuda_common.h"
#include "dense.cu"

// New global function for CuPy to call
extern "C" __global__ void
matmul_forward_cublaslt_kernel(floatX *out, floatX *inp, floatX *weight,
                               floatX *bias, int B, int T, int C, int OC) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    matmul_cublaslt(out, weight, inp, bias, OC, B * T, C, 0, true, false, 0, 0,
                    0, 0, false, false);
  }
}
