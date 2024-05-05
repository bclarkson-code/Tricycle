// Kernels for efficiently computing softmax derivatives
extern "C" {
__global__ void softmax_back_fn_1d(const float *softmax_result,
                                   const float *grad, const int n_elements,
                                   float *out) {
  int indicator, deriv;
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  int i = tid / n_elements;
  int j = tid % n_elements;

  if (i == j) {
    indicator = 1;
  } else {
    indicator = 0;
  }

  deriv = softmax_result[i] * (indicator - softmax_result[j]);
  out[j] = deriv * grad[i];
}
}
// __global__ void softmax_back_fn_2d(const float *softmax_result,
//                                    const float *grad, const int n_elements,
//                                    float *out) {
//   int indicator, i, j, deriv;
//
//   // index for vector
//   int t = blockDim.x * blockIdx.x + threadIdx.x;
//
//   // index for element in vector
//   int tid = blockDim.y * blockIdx.y + threadIdx.y;
//   i = tid / n_elements;
//   j = tid % n_elements;
//
//   if (i == j) {
//     indicator = 1;
//   } else {
//     indicator = 0;
//   }
//
//   deriv = softmax_result[t, i] * (indicator - softmax_result[t, j]);
//   out[t, j] = deriv * grad[t, i];
// }
//
// __global__ void softmax_autoregressive_backward_kernel2(
//     const float *grad, const float *softmax_result, int n_batches, int n_tokens,
//     int n_elements, float *out, ) {
//   int t3 = blockIdx.x * blockDim.x + threadIdx.x;
//   int idx = blockIdx.y * n_tokens * n_tokens;
//   if (t3 >= n_tokens) {
//     return;
//   }
//
//   for (int t = t3; t < n_tokens; t++) {
//     float result = 0.0;
//     const float *softmax_result_bth = softmax_result + idx + t * n_tokens;
//     const float *grad_bth = grad + idx + t * n_tokens;
//     float *out_bth = out + idx + t * n_tokens;
//
//     for (int t2 = 0; t2 <= t; t2++) {
//       float indicator = t2 == t3 ? 1.0f : 0.0f;
//       float local_derivative =
//           softmax_result_bth[t2] * (indicator - softmax_result_bth[t3]);
//       result += local_derivative * grad_bth[t2];
//     }
//
//     out_bth[t3] = result;
//   }
// }
// __global__ void softmax_back_fn_3d(const float *softmax_result,
//                                    const float *grad,
//
//                                    const int n_tokens, const int n_elements,
//                                    float *out) {
//   int indicator, i, j, b, t, deriv;
//
//   // find indices for batch and token
//   int xid = blockDim.x * blockIdx.x + threadIdx.x;
//   b = xid / n_tokens;
//   t = xid % n_tokens;
//
//   // index for element in vector
//   int tid = blockDim.y * blockIdx.y + threadIdx.y;
//   i = tid / n_elements;
//   j = tid % n_elements;
//
//   if (i == j) {
//     indicator = 1;
//   } else {
//     indicator = 0;
//   }
//
//   deriv = softmax_result[b, t, i] * (indicator - softmax_result[b, t, j]);
//   out[b, t, j] = deriv * grad[b, t, i];
// }
// __global__ void softmax_back_fn_4d(const float *softmax_result,
//                                    const float *grad, const int n_heads,
//                                    const int n_tokens, const int n_elements,
//                                    float *out) {
//   int indicator, i, j, b, h, t, remainder, deriv;
//
//   // find indices for batch and token
//   int xid = blockDim.x * blockIdx.x + threadIdx.x;
//   b = xid / (n_tokens * n_heads);
//   remainder = xid % (n_tokens * n_heads);
//   h = remainder / n_tokens;
//   t = remainder % n_tokens;
//
//   // index for element in vector
//   int tid = blockDim.y * blockIdx.y + threadIdx.y;
//   i = tid / n_elements;
//   j = tid % n_elements;
//
//   if (i == j) {
//     indicator = 1;
//   } else {
//     indicator = 0;
//   }
//
//   deriv = softmax_result[b, h, t, i] * (indicator - softmax_result[b, h, t, j]);
//   // sometimes this returns nans
//   out[b, h, t, j] = deriv * grad[b, h, t, i];
// }
// __global__ void softmax_back_fn_3d_a(const float *grad,
//                                      const float *softmax_result,
//                                      const int n_batches, const int n_tokens,
//                                      const int n_elements, float *out) {
//   int t3 = blockIdx.x * blockDim.x + threadIdx.x;
//   int idx = blockIdx.y * n_tokens * n_tokens;
//   if (t3 >= n_tokens) {
//     return;
//   }
//   for (int t = t3; t < n_tokens; t++) {
//     float result = 0.0;
//     const float *softmax_result_bth = softmax_result + idx + t * n_tokens;
//     const float *grad_bth = grad + idx + t * n_tokens;
//     float *out_bth = out + idx + t * n_tokens;
//     for (int t2 = 0; t2 <= t; t2++) {
//       float indicator = t2 == t3 ? 1.0f : 0.0f;
//       float local_derivative =
//           softmax_result_bth[t2] * (indicator - softmax_result_bth[t3]);
//       result += local_derivative * grad_bth[t2];
//     }
//     out_bth[t3] = result;
//   }
// }
// }
