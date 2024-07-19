#include "cuda_common.h"
#include "cuda_utils.cuh"

extern "C" {

#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)

__global__ void gelu_forward(floatX *output, const floatX *input) {
  int idx = (blockIdx.x * blockDim.x + threadIdx.x);

  float xi = input[idx];
  float cube = 0.044715f * xi * xi * xi;
  output[idx] =
      (floatX)(0.5f * xi * (1.0f + tanhf(GELU_SCALING_FACTOR * (xi + cube))));
}

__global__ void gelu_backward(floatX *grad, const floatX *input) {
  int idx = (blockIdx.x * blockDim.x + threadIdx.x);

  float x = input[idx];
  float cube = 0.044715f * x * x * x;
  float tanh_arg = GELU_SCALING_FACTOR * (x + cube);
  float tanh_out = tanhf(tanh_arg);
  float coshf_out = coshf(tanh_arg);
  float sech_out = 1.0f / (coshf_out * coshf_out);
  float local_grad =
      0.5f * (1.0f + tanh_out) + x * 0.5f * sech_out * GELU_SCALING_FACTOR *
                                     (1.0f + 3.0f * 0.044715f * x * x);
  grad[idx] = (floatX)(local_grad * (float)grad[idx]);
}
}
