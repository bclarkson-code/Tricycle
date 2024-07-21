#include "cuda_common.h"

extern "C" {

__global__ void relu_forward(const floatX *input, floatX *output,
                             bool *below_zero) {
  floatX ZERO = 0.0;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  below_zero[idx] = input[idx] <= ZERO;
  output[idx] = input[idx] <= ZERO ? ZERO : input[idx];
}

__global__ void relu_backward(const floatX *grad, floatX *output,
                              bool *below_zero) {
  floatX ZERO = 0.0;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  output[idx] = below_zero[idx] ? ZERO : grad[idx];
}
}
