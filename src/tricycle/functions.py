from pathlib import Path

import cupy as cp

from tricycle.binary import bdiv
from tricycle.reduce import rmax
from tricycle.tensor import Tensor, to_tensor
from tricycle.unary import udiv, uexp

softmax_kernel_path = Path(__file__).parent / "cuda/softmax.cu"
# module = cp.RawModule(path=str(softmax_kernel_path.absolute()))


softmax_1d_kernel = r"""
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
"""
softmax_3d_kernel = r"""
extern "C" __global__
void softmax_back_fn_3d(const float *softmax_result,
                                   const float *grad,

                                   const int n_tokens, const int n_elements,
                                   float *out) {
  int indicator, i, j, b, t, deriv;

  // find indices for batch and token
  int xid = blockDim.x * blockIdx.x + threadIdx.x;
  b = xid / n_tokens;
  t = xid % n_tokens;

  // index for element in vector
  int tid = blockDim.y * blockIdx.y + threadIdx.y;
  i = tid / n_elements;
  j = tid % n_elements;

  if (i == j) {
    indicator = 1;
  } else {
    indicator = 0;
  }

  deriv = softmax_result[b, t, i] * (indicator - softmax_result[b, t, j]);
  out[b, t, j] = deriv * grad[b, t, i];
}
"""
softmax_andrej_kernel = r"""
extern "C" __global__
void softmax_back_fn_4d_a(const float *grad,
                                     const float *softmax_result,
                                     const int n_batches, const int n_tokens,
                                     const int n_elements, float *out) {
  int t3 = blockIdx.x * blockDim.x + threadIdx.x;
  int idx = blockIdx.y * n_tokens * n_tokens;
  if (t3 >= n_tokens) {
    return;
  }
  for (int t = t3; t < n_tokens; t++) {
    float result = 0.0;
    const float *softmax_result_bth = softmax_result + idx + t * n_tokens;
    const float *grad_bth = grad + idx + t * n_tokens;
    float *out_bth = out + idx + t * n_tokens;
    for (int t2 = 0; t2 <= t; t2++) {
      float indicator = t2 == t3 ? 1.0f : 0.0f;
      float local_derivative =
          softmax_result_bth[t2] * (indicator - softmax_result_bth[t3]);
      result += local_derivative * grad_bth[t2];
    }
    out_bth[t3] = result;
  }
}
"""
softmax_back_fn_1d = cp.RawKernel(softmax_1d_kernel, "softmax_back_fn_1d")
softmax_back_fn_3d = cp.RawKernel(softmax_3d_kernel, "softmax_back_fn_3d")

softmax_back_fn_4d = cp.RawKernel(
    softmax_andrej_kernel, "softmax_back_fn_4d_a"
)


def _cuda_softmax_back_fn(grad, _result):
    import cupy as cp

    out = cp.zeros(_result.shape)
    _result = cp.asarray(_result)
    grad._data = cp.asarray(grad._data)
    n_elements = cp.int8(_result.shape[-1])

    BLOCK_SIZE = 32
    match _result.ndim:
        case 1:
            if _result.shape[0] % BLOCK_SIZE != 0:
                raise ValueError(
                    f"Expected shape to be divisible by {BLOCK_SIZE}, found: {_result.shape}"
                )
            grid_size = (_result.shape[0] // BLOCK_SIZE,)
            block_size = (BLOCK_SIZE,)
            softmax_back_fn_1d(
                grid_size,
                block_size,
                (_result, grad._data, n_elements, out),
            )
        case 2:
            if (
                _result.shape[0] % BLOCK_SIZE != 0
                and _result.shape[1] % BLOCK_SIZE != 0
            ):
                raise ValueError(
                    f"Expected shape to be divisible by {BLOCK_SIZE}, found: {_result.shape}"
                )

            grid_size = (
                _result.shape[0] // BLOCK_SIZE,
                _result.shape[1] // BLOCK_SIZE,
            )
            block_size = (BLOCK_SIZE, BLOCK_SIZE)
            softmax_back_fn_2d(
                grid_size,
                block_size,
                (_result, grad._data, n_elements, out),
            )
        case 3:
            n_batches, n_tokens, n_elements = _result.shape
            # for some reason, 3d blocks aren;t working for me so we'll use a
            # 2d one instead
            grid_size = (
                (n_batches * n_tokens) // BLOCK_SIZE,
                n_elements // BLOCK_SIZE,
            )
            block_size = (BLOCK_SIZE, BLOCK_SIZE)
            breakpoint()
            softmax_back_fn_3d(
                grid_size,
                block_size,
                (_result, grad._data, n_tokens, n_elements, out),
            )
        case 4:
            n_batches, n_heads, n_tokens, n_elements = _result.shape
            # for some reason, 3d blocks aren;t working for me so we'll use a
            # 2d one instead
            grid_size = (
                (n_batches * n_heads * n_tokens) // BLOCK_SIZE,
                n_elements // BLOCK_SIZE,
            )
            block_size = (BLOCK_SIZE, BLOCK_SIZE)
            breakpoint()
            softmax_back_fn_4d(
                grid_size,
                block_size,
                (grad._data, _result, n_batches, n_tokens, n_elements, out),
            )
        case _:
            raise NotImplementedError()
    # for reasons I dont understand, the cuda softmax sometimes returns nan
    # I think this might be a numerical precision thing but im not sure
    # for now, replacing the nans with 0's doesnt seem to hurt
    out = cp.nan_to_num(out, nan=0)
    out = to_tensor(out, is_vector=grad.is_vector, name="back_softmax")
    breakpoint()
    return out


def softmax_old(tensor: Tensor):
    """
    Apply softmax. The softmax is only applied to the final
    dimension of the tensor
    Note: the tensor is normalised for numeric stability
    """
    # add a really small number to the denominator to avoid infitiies
    REALLY_SMALL_NUMBER = 1e-8
    # normalise
    match tensor.ndim:
        case 1:
            largest_element = rmax(tensor, "a->").repeat(tensor.shape[-1])
        case _:
            largest_element = rmax(tensor, "...a->...").repeat(
                tensor.shape[-1]
            )
    tensor = tensor - largest_element

    numerator = uexp(tensor)
    denominator = numerator.e("...a->...")
    denominator += REALLY_SMALL_NUMBER
    denominator = denominator.repeat(tensor.shape[-1])

    return bdiv(numerator, denominator)


def softmax(tensor: Tensor):
    """
    Apply softmax. The softmax is only applied to the final
    dimension of the tensor
    Note: the tensor is normalised for numeric stability

    Note: This function is in development and not yet ready for use
    """

    if tensor.on_gpu:
        from cupyx.scipy.special import softmax as softmax_fn
    else:
        from scipy.special import softmax as softmax_fn

    _result = softmax_fn(tensor._data, axis=-1)

    def softmax_back_fn(grad):
        return _cuda_softmax_back_fn(grad, _result)

    result = to_tensor(_result)
    result.args = (tensor,)
    result.name = "softmax"
    result.is_vector = tensor.is_vector
    result.back_fns = (softmax_back_fn,)

    return result


def sigmoid(tensor: Tensor):
    """
    Apply the sigmoid function
    """
    return udiv(1, (uexp(-tensor) + 1))


def tanh(tensor: Tensor):
    """
    Apply the tanh function
    """
    numerator = uexp(tensor * 2) - 1
    denominator = uexp(tensor * 2) + 1
    return bdiv(numerator, denominator)
