#include "cuda_common.h"

extern "C" {

#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)

__global__ void gelu_forward(floatX *output, const floatX *input) {
  int idx = (blockIdx.x * blockDim.x + threadIdx.x);

  floatX x = input[idx];
  floatX cube = 0.044715f * x * x * x;
  output[idx] = 0.5f * x * (1.0f + tanhf(GELU_SCALING_FACTOR * (x + cube)));
}

__global__ void gelu_backward(floatX *grad, const floatX *input) {
  int idx = (blockIdx.x * blockDim.x + threadIdx.x);

  floatX x = input[idx];
  floatX cube = 0.044715f * x * x * x;
  floatX tanh_arg = GELU_SCALING_FACTOR * (x + cube);
  floatX tanh_out = tanhf(tanh_arg);
  floatX cosh_out = coshf(tanh_arg);
  floatX sech_out = 1.0f / (cosh_out * cosh_out);
  floatX local_grad =
      0.5f * (1.0f + tanh_out) + x * 0.5f * sech_out * GELU_SCALING_FACTOR *
                                     (1.0f + 3.0f * 0.044715f * x * x);
  grad[idx] = local_grad * grad[idx];
}
}
