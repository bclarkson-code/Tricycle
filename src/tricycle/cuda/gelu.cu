#include "cuda_common.h"
#include "cuda_utils.cuh"

extern "C" {

#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)
__global__ void gelu_forward(floatX *out, const floatX *inp) {
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;

  x128 packed_out;
  x128 packed_inp = load128cs(inp + idx); // load and do not keep in cache
  for (int k = 0; k < packed_inp.size; ++k) {
    float xi = (float)packed_inp[k];
    float cube = 0.044715f * xi * xi * xi;
    packed_out[k] =
        (floatX)(0.5f * xi * (1.0f + tanhf(GELU_SCALING_FACTOR * (xi + cube))));
  }
  // store instead of storecs (without cache streaming) in case it is useful for
  // the data to be in the cache for the next operation after this GeLU
  store128(out + idx, packed_out);
}

__global__ void gelu_backward(floatX *d_in_out, const floatX *inp) {
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;

  x128 packed_dinp;
  x128 packed_inp = load128cs(inp + idx);
  x128 packed_dout = load128(d_in_out + idx);
  for (int k = 0; k < packed_inp.size; ++k) {
    float x = (float)packed_inp[k];
    float cube = 0.044715f * x * x * x;
    float tanh_arg = GELU_SCALING_FACTOR * (x + cube);
    float tanh_out = tanhf(tanh_arg);
    float coshf_out = coshf(tanh_arg);
    float sech_out = 1.0f / (coshf_out * coshf_out);
    float local_grad =
        0.5f * (1.0f + tanh_out) + x * 0.5f * sech_out * GELU_SCALING_FACTOR *
                                       (1.0f + 3.0f * 0.044715f * x * x);
    packed_dinp[k] = (floatX)(local_grad * (float)packed_dout[k]);
  }
  store128(d_in_out + idx, packed_dinp);
}
}
