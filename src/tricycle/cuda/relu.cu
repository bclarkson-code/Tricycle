extern "C"{

__global__ void relu_forward(const float *input, float *output, bool *below_zero) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    below_zero[idx] = input[idx] <= 0;
    output[idx] = below_zero[idx] ? 0 : input[idx];
}

__global__ void relu_backward(const float *grad, float *output, bool *below_zero) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    output[idx] = below_zero[idx] ? 0 : grad[idx];
}

}
