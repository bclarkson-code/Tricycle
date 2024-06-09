from abc import ABC, abstractmethod
from typing import Sequence

from tricycle.binary import BinaryMultiply
from tricycle.initialisers import init_xavier
from tricycle.optimisers import Optimiser
from tricycle.tensor import Tensor, nothing, to_tensor


class Layer(ABC):
    tensors: dict[str, Tensor] = {}
    layers: Sequence["Layer"] = []

    @abstractmethod
    def forward(self, tensor: Tensor):
        raise NotImplementedError

    def __call__(self, tensor: Tensor, *args, **kwargs):
        return self.forward(tensor, *args, **kwargs)

    def update(self, optimiser: Optimiser):
        pass

    def zero_grad(self):
        pass

    def to_gpu(self, device: int = 0):
        pass

    def from_gpu(self):
        pass


class Dense(Layer):
    weights: Tensor
    from_size: int
    to_size: int
    name: str | None

    def __init__(
        self, from_size: int, to_size: int, initialiser=init_xavier, name=None
    ):
        self.weights = initialiser(
            (from_size, to_size), name="weights" if name is None else name
        )
        self.name = name
        self.from_size = from_size
        self.to_size = to_size
        self.tensors = {"weights": self.weights}

    def weight_back_fn(self, grad: Tensor):
        xp = grad.xp
        result = xp.einsum(self._weight_subscript, self._input, grad.array)
        return to_tensor(
            result,
            requires_grad=grad.requires_grad,
            name="back_dense",
            is_vector=False,
        )

    def grad_back_fn(self, grad: Tensor):
        xp = grad.xp
        result = xp.einsum(
            self._grad_subscript, self.weights.array, grad.array
        )
        return to_tensor(
            result,
            requires_grad=grad.requires_grad,
            name="back_dense",
            is_vector=True,
        )

    def forward(self, tensor: Tensor):
        match tensor.ndim:
            case 1:
                subscript = "a,aW->W"
                weight_subscript = "a,W->aW"
                grad_subscript = "aW,W->a"
            case 2:
                subscript = "Tb,bW->TW"
                weight_subscript = "Tb,TW->bW"
                grad_subscript = "bW,TW->Tb"
            case 3:
                subscript = "zTb,bW->zTW"
                weight_subscript = "zTb,zTW->bW"
                grad_subscript = "bW,zTW->zTb"
            case 4:
                subscript = "zxTb,bW->zxTW"
                weight_subscript = "zxTb,zxTW->bW"
                grad_subscript = "bW,zxTW->zxTb"
            case _:
                raise NotImplementedError(
                    f"Cannot pass tensor with shape {tensor.shape} "
                    f"and {tensor.is_vector=}"
                    "through a Dense layer"
                )
        result = to_tensor(
            tensor.xp.einsum(
                subscript,
                tensor.array,
                self.weights.array,
            )
        )
        self._grad_subscript = grad_subscript
        self._weight_subscript = weight_subscript
        self._input = tensor.array

        result.name = "dense"
        result.args = (self.weights, tensor)
        result.back_fns = (self.weight_back_fn, self.grad_back_fn)
        result.is_vector = tensor.is_vector

        return result

    def update(self, optimiser: Optimiser):
        self.weights = optimiser(self.weights)

    def zero_grad(self):
        self.weights.grad = None

    def to_gpu(self, device: int = 0):
        self.weights.to_gpu(device)
        return self

    def from_gpu(self):
        self.weights.from_gpu()
        return self


# class Dropout(Layer):
#     def __init__(self, probability: float):
#         self.probability = probability
#
#     def backward(self, grad: Tensor):
#         return to_tensor(self._mask * grad._data, is_vector=grad.is_vector)
#
#     def forward(self, tensor: Tensor):
#         if self.probability == 0:
#             return tensor
#         xp = tensor.xp
#         coef = 1 / (1 - self.probability)
#
#         self._mask = (xp.random.rand(*tensor.shape) > self.probability).astype(
#             tensor.dtype
#         ) * coef
#         self._out = self._mask * tensor._data
#         result = to_tensor(self._out, is_vector=tensor.is_vector)
#         result.args = (tensor,)
#         result.back_fns = (self.backward,)
#
#         return result


class Dropout(Layer):
    def __init__(self, probability: float):
        self.probability = probability

    def forward(self, tensor: Tensor):
        if self.probability == 0:
            return tensor
        xp = tensor.xp
        coef = 1 / (1 - self.probability)
        random_mask = (
            xp.random.rand(*tensor.shape) > self.probability
        ).astype(tensor.dtype) * coef
        random_mask = to_tensor(
            random_mask, is_vector=True, requires_grad=False
        )
        return BinaryMultiply()(tensor, random_mask)


class LayerNorm(Layer):
    def __init__(self, embedding_dim: int, eps=1e-5):
        import numpy as np

        self.eps = eps
        self.gamma = to_tensor(
            np.ones((embedding_dim,)), requires_grad=True, is_vector=False
        )
        self.beta = to_tensor(
            np.zeros((embedding_dim,)), requires_grad=True, is_vector=False
        )

    def forward(self, tensor: Tensor):
        """
        Performs Layer Normalization on the input tensor x.

        Args:
            x (numpy.ndarray): Input tensor of shape (batch_size, *).

        Returns:
            numpy.ndarray: Normalized tensor of the same shape as x.
        """
        xp = tensor.xp
        x = tensor.array

        # Compute mean and variance along the feature dimension
        self._mean = x.mean(axis=-1, keepdims=True)
        self._var = x.var(axis=-1, keepdims=True)
        self._input = x

        # Normalize and scale
        x_norm = (x - self._mean) / xp.sqrt(self._var + self.eps)
        output = self.gamma.array * x_norm + self.beta.array

        output = to_tensor(
            output,
            is_vector=tensor.is_vector,
            requires_grad=tensor.requires_grad,
        )
        output.back_fns = (self.back_fn, self.beta_back_fn, self.gamma_back_fn)
        output.args = (tensor, self.beta, self.gamma)
        output.name = "layer_norm"

        return output

    def gamma_back_fn(self, grad: Tensor):
        """
        backward pass for grad
        """
        xp = grad.xp

        # Compute intermediate values
        x_norm = (self._input - self._mean) / xp.sqrt(self._var + self.eps)
        axes = tuple(range(grad.ndim - 1))
        result = xp.sum(grad.array * x_norm, axis=axes)
        return to_tensor(result, is_vector=False)

    def beta_back_fn(self, grad: Tensor):
        """
        backward pass for grad
        """
        xp = grad.xp

        # Compute intermediate values
        axes = tuple(range(grad.ndim - 1))
        result = xp.sum(grad.array, axis=axes)
        return to_tensor(result, is_vector=False)

    def back_fn(self, grad: Tensor):
        """
        backward pass for grad
        """
        xp = grad.xp

        # Compute intermediate values
        n = self._input.shape[-1]

        # Gradients with respect to x
        dx_norm = grad.array * self.gamma.array
        dvar = xp.sum(
            dx_norm
            * (self._input - self._mean)
            * -0.5
            * xp.power(self._var + self.eps, -1.5),
            axis=-1,
            keepdims=True,
        )
        dmean = xp.sum(
            dx_norm * -1 / xp.sqrt(self._var + self.eps),
            axis=-1,
            keepdims=True,
        ) + dvar * xp.mean(
            -2 * (self._input - self._mean) / n, axis=-1, keepdims=True
        )
        result = (
            dx_norm / xp.sqrt(self._var + self.eps)
            + dvar * 2 * (self._input - self._mean) / n
            + dmean / n
        )

        return to_tensor(
            result,
            is_vector=grad.is_vector,
            requires_grad=grad.requires_grad,
            name="back_ln",
        )

    def update(self, optimiser: Optimiser):
        self.gamma = optimiser(self.gamma)
        self.beta = optimiser(self.beta)

    def zero_grad(self):
        self.gamma.grad = None
        self.beta.grad = None

    def to_gpu(self, device: int = 0):
        self.gamma.to_gpu(device)
        self.beta.to_gpu(device)
        return self

    def from_gpu(self):
        self.gamma.from_gpu()
        self.beta.from_gpu()
        return self


class RMSNorm(Layer):
    """
    Normalise tensors by their sum of squares. This is similar to layer norm
    but removes means
    """

    REALLY_SMALL_NUMBER = 1e-6

    def __init__(self, to_size: int):
        raise NotImplementedError(
            "RMSNorm is still in development and not ready for use"
        )
        import numpy as np

        self.weights = to_tensor(np.ones(to_size))

    def build_back_fn(self, rms, input_):
        def rmsnorm_weight_back_fn(grad):
            xp = grad.xp
            result = xp.sum(input_ / rms, axis=-2).sum(0).squeeze()
            return to_tensor(result, is_vector=False)

        def rmsnorm_back_fn(grad):
            xp = grad.xp
            scaled_grad = xp.multiply(grad._data, self.weights.array)

            left = scaled_grad / rms

            coef = scaled_grad / (384 * rms**3)

            match input_.ndim:
                case 2:
                    square_prod = xp.einsum("AB,Ac->AB", input_, input_)
                case 3:
                    square_prod = xp.einsum("zAB,zAc->zAB", input_, input_)
                case 4:
                    square_prod = xp.einsum("zxAB,zxAc->zxAB", input_, input_)
                case _:
                    raise NotImplementedError(
                        f"RMSNorm with tensors of size {input_.ndim} are not yet supported"
                    )
            right = square_prod * coef
            return to_tensor(left - right, is_vector=grad.is_vector)

        return rmsnorm_weight_back_fn, rmsnorm_back_fn

    def forward(self, tensor: Tensor):
        xp = tensor.xp
        square_sum = (tensor.array * tensor.array).mean(axis=-1)
        rms = xp.sqrt(square_sum)
        rms = xp.expand_dims(rms, -1)
        result = xp.divide(tensor.array, (rms + self.REALLY_SMALL_NUMBER))
        result = xp.einsum("...a,a->...a", result, self.weights.array)

        weight_back_fn, back_fn = self.build_back_fn(
            rms=rms, input_=tensor.array, is_vector=tensor.is_vector
        )
        result = to_tensor(result, is_vector=tensor.is_vector, name="rmsnorm")
        result.back_fns = (
            weight_back_fn,
            back_fn,
        )
        result.args = (
            self.weights,
            tensor,
        )

        return result

    def update(self, optimiser: Optimiser):
        self.weights = optimiser(self.weights)

    def zero_grad(self):
        self.weights.grad = None

    def to_gpu(self, device: int = 0):
        self.weights.to_gpu(device)
        return self

    def from_gpu(self):
        self.weights.from_gpu()
        return self


class Embedding(Layer):
    """
    Convert an index to an embedding with a lookup (rather than a one-hot
    encoding and a matrix multiplication)
    """

    def __init__(
        self,
        from_size: int,
        to_size: int,
        name: str | None = None,
        initialiser=init_xavier,
    ):
        self.weights = initialiser(
            (from_size, to_size), name=name or "weights"
        )
        self.vocab_size = from_size

    def back_fn(self, grad: Tensor):
        xp = grad.xp
        out = xp.zeros(self.weights.shape)

        match grad.ndim - self.input.ndim:
            case 1:
                xp.add.at(out, self.input.array, grad.array)
            case 2:
                xp.add.at(out, self.input.array, grad.array.sum(axis=0))
            case _:
                raise NotImplementedError(
                    f"{grad.ndim=}, {self.input.ndim=} are not supported"
                )

        return to_tensor(out)

    def forward(self, tensor: Tensor):
        assert (
            tensor.requires_grad is False
        ), "Cannot embed a differentiable tensor"

        self.input = tensor
        if tensor.is_vector:
            self._out = self.weights.array[tensor.array.flatten()].reshape(
                tensor.array.shape + (-1,)
            )
        else:
            self._out = self.weights.array[tensor.array]
        result = to_tensor(self._out, is_vector=tensor.is_vector)

        result.args = (tensor, self.weights)

        result.back_fns = (nothing, self.back_fn)
        return result

    def update(self, optimiser: Optimiser):
        self.weights = optimiser(self.weights)

    def zero_grad(self):
        self.weights.grad = None

    def to_gpu(self, device: int = 0):
        self.weights.to_gpu(device)
        return self

    def from_gpu(self):
        self.weights.from_gpu()
        return self


class Sequential(Layer):
    def __init__(self, *layers: Layer):
        self.layers = layers

    def __getitem__(self, idx):
        return self.layers[idx]

    def forward(self, tensor: Tensor):
        for layer in self.layers:
            tensor = layer(tensor)
        return tensor

    def update(self, optimiser: Optimiser):
        for layer in self.layers:
            layer.update(optimiser)

    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()

    def to_gpu(self, device: int = 0):
        for layer in self.layers:
            layer.to_gpu(device)

    def from_gpu(self):
        for layer in self.layers:
            layer.from_gpu()
