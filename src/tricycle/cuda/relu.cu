extern "C"{

__global__ void relu_forward(const float *input, float output,
                             bool below_zero) {

  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * float ::size;

  below_zero[idx] = input[idx] <= 0;
  ouput[idx] = below_zero[idx] == true ? 0 : input[idx];
}

}
