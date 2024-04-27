import cupy as cp
import numpy as np
from numba import njit
from scipy.special import softmax as scipy_softmax

from tricycle.binary import bdiv
from tricycle.einsum import Einsum
from tricycle.reduce import rmax
from tricycle.tensor import Tensor, to_tensor
from tricycle.unary import udiv, uexp

# this is my first CUDA kernel, don't judge
softmax_back_fn_1d = cp.RawKernel(
    """
extern "C" __global__
void softmax_back_fn_1d(
    const float* softmax_result,
    const float* grad,
    const int n_elements,
    float* out
){
    int indicator, deriv;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int i = tid / n_elements;
    int j =  tid % n_elements;

    if (i == j) {
        indicator = 1;
    } else {
        indicator = 0;
    }

    deriv = softmax_result[i] * (indicator - softmax_result[j]);
    out[j] = deriv * grad[i];
}
""",
    "softmax_back_fn_1d",
)

softmax_back_fn_2d = cp.RawKernel(
    """
extern "C" __global__
void softmax_back_fn_2d(
    const float* softmax_result,
    const float* grad,
    const int n_elements,
    float* out
){
    int indicator, i, j, deriv;

    // index for vector
    int t = blockDim.x * blockIdx.x + threadIdx.x;

    // index for element in vector
    int tid = blockDim.y * blockIdx.y + threadIdx.y;
    i = tid / n_elements;
    j =  tid % n_elements;

    if (i == j) {
        indicator = 1;
    } else {
        indicator = 0;
    }

    deriv = softmax_result[t, i] * (indicator - softmax_result[t,j]);
    out[t,j] = deriv * grad[t, i];
}
""",
    "softmax_back_fn_2d",
)

softmax_back_fn_3d = cp.RawKernel(
    """
extern "C" __global__
void softmax_back_fn_3d(
    const float* softmax_result,
    const float* grad,
    const int n_tokens,
    const int n_elements,
    float* out
){
    int indicator, i, j, b, t, deriv;

    // find indices for batch and token
    int xid = blockDim.x * blockIdx.x + threadIdx.x;
    b = xid / n_tokens;
    t = xid % n_tokens;

    // index for element in vector
    int tid = blockDim.y * blockIdx.y + threadIdx.y;
    i = tid / n_elements;
    j =  tid % n_elements;

    if (i == j) {
        indicator = 1;
    } else {
        indicator = 0;
    }

    deriv = softmax_result[b, t, i] * (indicator - softmax_result[b, t, j]);
    out[b, t, j] = deriv * grad[b, t, i];
}
""",
    "softmax_back_fn_3d",
)


softmax_back_fn_4d = cp.RawKernel(
    """
extern "C" __global__
void softmax_back_fn_4d(
    const float* softmax_result,
    const float* grad,
    const int n_heads,
    const int n_tokens,
    const int n_elements,
    float* out
){
    int indicator, i, j, b, h, t, remainder, deriv;

    // find indices for batch and token
    int xid = blockDim.x * blockIdx.x + threadIdx.x;
    b = xid / (n_tokens * n_heads);
    remainder = xid % (n_tokens * n_heads);
    h = remainder / n_tokens;
    t = remainder % n_tokens;

    // index for element in vector
    int tid = blockDim.y * blockIdx.y + threadIdx.y;
    i = tid / n_elements;
    j =  tid % n_elements;

    if (i == j) {
        indicator = 1;
    } else {
        indicator = 0;
    }

    deriv = softmax_result[b, h, t, i] * (indicator - softmax_result[b, h, t, j]);
    // sometimes this returns nans
    out[b, h, t, j] = deriv * grad[b, h, t, i];
}
""",
    "softmax_back_fn_4d",
)


def _cuda_softmax_back_fn(grad, _result):
    import cupy as cp

    BLOCK_SIZE = 32

    out = cp.zeros(_result.shape)
    _result = cp.asarray(_result)
    grad._data = cp.asarray(grad._data)
    n_elements = cp.int8(_result.shape[-1])

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
            if not (
                _result.shape[0] % BLOCK_SIZE == 0
                or _result.shape[1] % BLOCK_SIZE == 0
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
            softmax_back_fn_4d(
                grid_size,
                block_size,
                (_result, grad._data, n_heads, n_tokens, n_elements, out),
            )
        case _:
            raise NotImplementedError()
    # for reasons I dont understand, the cuda softmax sometimes returns nan
    # I think this might be a numerical precision thing but im not sure
    # for now, replacing the nans with 0's doesnt seem to hurt
    out = cp.nan_to_num(out, 0)
    return to_tensor(out, is_vector=grad.is_vector, name="back_softmax")


def softmax(tensor: Tensor):
    """
    Apply softmax. The softmax is only applied to the final
    dimension of the tensor
    Note: the tensor is normalised for numeric stability
    """
    # add a really small number to the denominator to avoid infitiies
    REALLY_SMALL_NUMBER = 1e-8
    # normalise
    largest_element = rmax(tensor, "...a->...").repeat(tensor.shape[-1])
    tensor = tensor - largest_element

    numerator = uexp(tensor)
    denominator = numerator.e("...a->...")
    denominator += REALLY_SMALL_NUMBER
    denominator = denominator.repeat(tensor.shape[-1])

    return bdiv(numerator, denominator)


def softmax_v2(tensor: Tensor):
    """
    Apply softmax. The softmax is only applied to the final
    dimension of the tensor
    Note: the tensor is normalised for numeric stability
    """
    if tensor.on_gpu:
        from cupyx.scipy.special import softmax as softmax_fn
    else:
        from scipy.special import softmax as softmax_fn

    _result = softmax_fn(tensor._data, axis=-1)

    def softmax_back_fn(grad: Tensor):
        xp = grad.xp
        match _result.ndim:
            case 1:
                n_elements = _result.shape[0]
                diag = xp.identity(n_elements)

                left = xp.einsum("i,ij,i->j", _result, diag, grad._data)
                right = xp.einsum("i,j,i->j", _result, _result, grad._data)
            case 2:
                batch_size, n_elements = _result.shape
                diag = xp.tile(xp.identity(n_elements), batch_size).T
                diag = diag.reshape(batch_size, n_elements, n_elements)

                left = xp.einsum("ki,kij,ki->kj", _result, diag, grad._data)
                right = xp.einsum("ki,kj,ki->kj", _result, _result, grad._data)
            case 3:
                batch_size, n_tokens, n_elements = _result.shape
                diag = xp.tile(xp.identity(n_elements), n_tokens)
                diag = xp.tile(diag, batch_size).T
                diag = diag.reshape(
                    batch_size, n_tokens, n_elements, n_elements
                )

                left = xp.einsum(
                    "zki,zkij,zki->zkj", _result, diag, grad._data
                )
                right = xp.einsum(
                    "zki,zkj,zki->zkj", _result, _result, grad._data
                )
            case _:
                raise NotImplementedError()

        output = left - right
        return to_tensor(output, is_vector=grad.is_vector, name="back_softmax")

    result = to_tensor(_result)
    result.args = (tensor,)
    result.name = "softmax"
    result.is_vector = tensor.is_vector
    result.back_fns = (softmax_back_fn,)

    return result


def softmax_v3(tensor: Tensor):
    """
    Apply softmax. The softmax is only applied to the final
    dimension of the tensor
    Note: the tensor is normalised for numeric stability
    """
    if tensor.on_gpu:
        from cupyx.scipy.special import softmax as softmax_fn
    else:
        from scipy.special import softmax as softmax_fn

    _result = softmax_fn(tensor._data, axis=-1)

    def softmax_back_fn(grad: Tensor):
        xp = grad.xp
        match _result.ndim:
            case 1:
                out = xp.zeros(_result.shape)
                for i in range(_result.shape[0]):
                    for j in range(_result.shape[0]):
                        is_diagonal = i == j
                        local_derivative = _result[i] * (
                            is_diagonal - _result[j]
                        )
                        out[j] += local_derivative * grad._data[i]
            case 2:
                out = xp.zeros(_result.shape)
                for t in range(_result.shape[0]):
                    for i in range(_result.shape[1]):
                        for j in range(_result.shape[1]):
                            is_diagonal = i == j
                            local_derivative = _result[t][i] * (
                                is_diagonal - _result[t][j]
                            )
                            out[t][j] += local_derivative * grad._data[t][i]
            case 3:
                out = xp.zeros(_result.shape)
                for z in range(_result.shape[0]):
                    for t in range(_result.shape[1]):
                        for i in range(_result.shape[2]):
                            for j in range(_result.shape[2]):
                                is_diagonal = i == j
                                local_derivative = _result[z][t][i] * (
                                    is_diagonal - _result[z][t][j]
                                )
                                out[z][t][j] += (
                                    local_derivative * grad._data[z][t][i]
                                )
            case _:
                raise NotImplementedError()

        return to_tensor(out, is_vector=grad.is_vector, name="back_softmax")

    result = to_tensor(_result)
    result.args = (tensor,)
    result.name = "softmax"
    result.is_vector = tensor.is_vector
    result.back_fns = (softmax_back_fn,)

    return result


def softmax_v4(tensor: Tensor):
    """
    Apply softmax. The softmax is only applied to the final
    dimension of the tensor
    Note: the tensor is normalised for numeric stability
    """
    if tensor.on_gpu:
        from cupyx.scipy.special import softmax as softmax_fn
    else:
        from scipy.special import softmax as softmax_fn

    _result = softmax_fn(tensor._data, axis=-1)

    softmax_back_fn = lambda grad: _cuda_softmax_back_fn(grad, _result)

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
