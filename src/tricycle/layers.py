from abc import ABC, abstractmethod
from string import ascii_letters
from typing import Sequence

from tricycle.binary import bmask, bmul
from tricycle.einsum import Einsum
from tricycle.initialisers import init_xavier
from tricycle.optimisers import Optimiser
from tricycle.tensor import Tensor, nothing, to_tensor
from tricycle.unary import usqrt


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
        self.from_size = from_size
        self.to_size = to_size
        self.tensors = {"weights": self.weights}

    def _build_missing_indices(
        self, tensor: Tensor, initial_subscript: str
    ) -> str:
        """
        In some circumstances, using ellipses with vectorised tensors
        can be defined in the forward direction but not in reverse.

        To fix this, we're building a string of indices that can be used
        in place of an ellipsis. This is a bit of an ugly hack, but it
        works for now.

        TODO: fix this properly
        """
        n_untouched_indices = (
            len(tensor.shape) - 2
            if tensor.is_vector
            else len(tensor.shape) - 1
        )
        untouched_indices = ""
        i = 0
        while len(untouched_indices) < n_untouched_indices:
            next_idx = ascii_letters[i]
            if (
                next_idx not in untouched_indices
                and next_idx != "z"
                and next_idx not in initial_subscript
            ):
                untouched_indices += next_idx
            i += 1
        return untouched_indices

    def forward(self, tensor: Tensor):
        initial_subscript = "a,aB->B"
        idx = self._build_missing_indices(tensor, initial_subscript)
        return Einsum(f"{idx}a,aB->{idx}B")(tensor, self.weights)

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


class DenseV2(Layer):
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
        self.from_size = from_size
        self.to_size = to_size
        self.tensors = {"weights": self.weights}

    def _build_missing_indices(
        self, tensor: Tensor, initial_subscript: str
    ) -> str:
        """
        In some circumstances, using ellipses with vectorised tensors
        can be defined in the forward direction but not in reverse.

        To fix this, we're building a string of indices that can be used
        in place of an ellipsis. This is a bit of an ugly hack, but it
        works for now.

        TODO: fix this properly
        """
        n_untouched_indices = (
            len(tensor.shape) - 2
            if tensor.is_vector
            else len(tensor.shape) - 1
        )
        untouched_indices = ""
        i = 0
        while len(untouched_indices) < n_untouched_indices:
            next_idx = ascii_letters[i]
            if (
                next_idx not in untouched_indices
                and next_idx != "z"
                and next_idx not in initial_subscript
            ):
                untouched_indices += next_idx
            i += 1
        return untouched_indices

    def forward(self, tensor: Tensor):
        initial_subscript = "a,aB->B"
        idx = self._build_missing_indices(tensor, initial_subscript)
        return Einsum(f"{idx}a,aB->{idx}B")(tensor, self.weights)

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


class DenseV3(Layer):
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

    def _einsum_fn(self, subscript, tensor, is_vector: bool):
        def back_einsum(grad):
            result = tensor.xp.einsum(subscript, tensor._data, grad._data)
            return to_tensor(
                result,
                requires_grad=grad.requires_grad,
                name="back_dense",
                is_vector=is_vector,
            )

        return back_einsum

    def forward(self, tensor: Tensor):
        match tensor.ndim:
            case 1:
                result = self._forward(tensor, "a,aW->W", "a,W->aW", "aW,W->a")
            case 2:
                result = self._forward(
                    tensor, "Tb,bW->TW", "Tb,TW->bW", "bW,TW->Tb"
                )
            case 3:
                result = self._forward(
                    tensor,
                    "zTb,bW->zTW",
                    "zTb,zTW->bW",
                    "bW,zTW->zTb",
                )
            case 4:
                result = self._forward(
                    tensor,
                    "zxTb,bW->zxTW",
                    "zxTb,zxTW->bW",
                    "bW,zxTW->zxTb",
                )
            case _:
                raise NotImplementedError(
                    f"Cannot pass tensor with shape {tensor.shape} "
                    f"and {tensor.is_vector=}"
                    "through a Dense layer"
                )
        result.name = "dense"
        return result

    def _forward(
        self,
        tensor,
        subscript,
        weight_subscript,
        tensor_subscript,
    ):
        result = to_tensor(
            tensor.xp.einsum(
                subscript,
                tensor._data,
                self.weights._data,
            )
        )
        weight_back_fn = self._einsum_fn(
            weight_subscript, tensor, is_vector=False
        )
        grad_back_fn = self._einsum_fn(
            tensor_subscript, self.weights, is_vector=tensor.is_vector
        )
        result.args = (self.weights, tensor)
        result.back_fns = (weight_back_fn, grad_back_fn)
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


class DenseV4(Layer):
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
        self.from_size = from_size
        self.to_size = to_size
        self.tensors = {"weights": self.weights}

    def forward(self, tensor: Tensor):
        result = to_tensor(tensor.xp.matmul(tensor._data, self.weights._data))

        def weight_back_fn(grad):
            result = tensor.xp.matmul(
                tensor._data.transpose(0, 2, 1), grad._data
            )
            return to_tensor(
                result,
                requires_grad=tensor.requires_grad,
                name="back_dense",
            )

        def input_back_fn(grad):
            result = tensor.xp.matmul(
                grad._data, self.weights._data.transpose()
            )
            return to_tensor(
                result,
                requires_grad=tensor.requires_grad,
                name="back_dense",
            )

        result.args = (self.weights, tensor)
        result.back_fns = (weight_back_fn, input_back_fn)
        result.is_vector = tensor.is_vector
        result.name = "dense"

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


class Dropout(Layer):
    def __init__(self, probability: float):
        self.probability = probability

    def forward(self, tensor: Tensor):
        random_mask = tensor.xp.random.binomial(
            n=1, p=1 - self.probability, size=tensor.shape
        )
        random_mask = to_tensor(
            random_mask, requires_grad=False, is_vector=tensor.is_vector
        )
        return bmul(tensor, random_mask)


class DropoutV2(Layer):
    def __init__(self, probability: float):
        self.probability = probability

    def forward(self, tensor: Tensor):
        random_mask = tensor.xp.random.choice(
            [True, False],
            size=tensor.shape,
            p=[self.probability, 1 - self.probability],
        )
        random_mask = to_tensor(
            random_mask, requires_grad=False, is_vector=tensor.is_vector
        )
        return bmul(tensor, random_mask)


class DropoutV3(Layer):
    def __init__(self, probability: float):
        self.probability = probability

    def forward(self, tensor: Tensor):
        random_mask = tensor.xp.random.binomial(
            n=1, p=1 - self.probability, size=tensor.shape
        )
        random_mask = to_tensor(
            random_mask, requires_grad=False, is_vector=tensor.is_vector
        )
        return bmask(tensor, random_mask)


class DropoutV4(Layer):
    def __init__(self, probability: float):
        self.probability = probability

    def forward(self, tensor: Tensor):
        random_mask = to_tensor(
            tensor.xp.random.choice(
                [True, False],
                size=tensor.shape,
                p=[self.probability, 1 - self.probability],
            ),
            requires_grad=False,
            is_vector=tensor.is_vector,
        )
        return bmask(tensor, random_mask)


class DropoutV5(Layer):
    def __init__(self, probability: float):
        self.probability = probability

    def forward(self, tensor: Tensor):
        shape = tensor.shape[1:] if tensor.is_vector else tensor.shape
        random_mask = tensor.xp.random.binomial(
            n=1, p=1 - self.probability, size=shape
        )
        random_mask = to_tensor(random_mask, requires_grad=False)
        return bmul(tensor, random_mask)


class DropoutV6(Layer):
    def __init__(self, probability: float):
        self.probability = probability

    def forward(self, tensor: Tensor):
        random_mask = tensor.xp.random.binomial(
            n=1, p=1 - self.probability, size=tensor.shape
        ).astype(bool)
        random_mask = to_tensor(
            random_mask, requires_grad=False, is_vector=tensor.is_vector
        )
        return bmul(tensor, random_mask)


class DropoutV7(Layer):
    def __init__(self, probability: float):
        self.probability = probability

    def forward(self, tensor: Tensor):
        if self.probability == 0:
            return tensor
        shape = tensor.shape[1:] if tensor.is_vector else tensor.shape
        random_mask = tensor.xp.random.binomial(
            n=1, p=1 - self.probability, size=shape
        ).astype(bool)
        random_mask = to_tensor(random_mask, requires_grad=False)
        return bmul(tensor, random_mask)


class LayerNorm(Layer):
    """
    Normalise each tensor individually
    """

    def forward(self, tensor: Tensor):
        return tensor.normalise()


class LayerNormV2(Layer):
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
        x = tensor._data

        # Compute mean and variance along the feature dimension
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)

        # Normalize and scale
        x_norm = (x - mean) / xp.sqrt(var + self.eps)
        output = self.gamma._data * x_norm + self.beta._data

        self.cache = x, mean, var

        output = to_tensor(
            output,
            is_vector=tensor.is_vector,
            requires_grad=tensor.requires_grad,
        )
        back_fn, beta_back_fn, gamma_back_fn = self.build_back_fns(
            x, mean, var
        )
        output.back_fns = (back_fn, beta_back_fn, gamma_back_fn)
        output.args = (tensor, self.beta, self.gamma)
        output.name = "layer_norm"

        return output

    def build_back_fns(self, x, mean, var):
        def gamma_back_fn(grad: Tensor):
            """
            backward pass for grad
            """
            xp = grad.xp

            # Compute intermediate values
            x_norm = (x - mean) / xp.sqrt(var + self.eps)
            axes = list(range(grad.ndim - 1))
            result = xp.sum(grad._data * x_norm, axis=axes)
            return to_tensor(result, is_vector=False)

        def beta_back_fn(grad: Tensor):
            """
            backward pass for grad
            """
            xp = grad.xp

            # Compute intermediate values
            axes = list(range(grad.ndim - 1))
            result = xp.sum(grad._data, axis=axes)
            return to_tensor(result, is_vector=False)

        def back_fn(grad: Tensor):
            """
            backward pass for grad
            """
            xp = grad.xp

            # Compute intermediate values
            n = x.shape[-1]

            # Gradients with respect to x
            dx_norm = grad._data * self.gamma._data
            dvar = xp.sum(
                dx_norm * (x - mean) * -0.5 * xp.power(var + self.eps, -1.5),
                axis=-1,
                keepdims=True,
            )
            dmean = xp.sum(
                dx_norm * -1 / xp.sqrt(var + self.eps), axis=-1, keepdims=True
            ) + dvar * xp.mean(-2 * (x - mean) / n, axis=-1, keepdims=True)
            result = (
                dx_norm / xp.sqrt(var + self.eps)
                + dvar * 2 * (x - mean) / n
                + dmean / n
            )

            return to_tensor(
                result,
                is_vector=grad.is_vector,
                requires_grad=grad.requires_grad,
                name="back_ln",
            )

        return back_fn, beta_back_fn, gamma_back_fn

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

    def forward(self, tensor: Tensor):
        divisor = usqrt((tensor**2).mean()).repeat(tensor.shape[-1])
        return tensor / divisor


class RMSNormV2(Layer):
    """
    Normalise tensors by their sum of squares. This is similar to layer norm
    but removes means
    """

    REALLY_SMALL_NUMBER = 1e-6

    def __init__(self, to_size: int):
        import numpy as np

        self.weights = to_tensor(np.ones(to_size))

    def build_back_fn(self, rms, input_, is_vector=False):
        def rmsnorm_weight_back_fn(grad):
            xp = grad.xp
            result = xp.sum(input_ / rms, axis=-2).sum(0).squeeze()
            return to_tensor(result, is_vector=False)

        def rmsnorm_back_fn(grad):
            xp = grad.xp
            scaled_grad = xp.multiply(grad._data, self.weights._data)

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
            # square_prod = xp.power(input_, 2) / self.weights.shape[-1]

            right = square_prod * coef
            return to_tensor(left - right, is_vector=grad.is_vector)

        return rmsnorm_weight_back_fn, rmsnorm_back_fn

    def forward(self, tensor: Tensor):
        xp = tensor.xp
        square_sum = (tensor._data * tensor._data).mean(axis=-1)
        rms = xp.sqrt(square_sum)
        rms = xp.expand_dims(rms, -1)
        result = xp.divide(tensor._data, (rms + self.REALLY_SMALL_NUMBER))
        result = xp.einsum("...a,a->...a", result, self.weights._data)

        weight_back_fn, back_fn = self.build_back_fn(
            rms=rms, input_=tensor._data, is_vector=tensor.is_vector
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


class DEPRACATED_EMBEDDING(Layer):
    """
    Convert an index to an embedding with a lookup (rather than a one-hot
    encoding and a matrix multiplication)
    """

    def __init__(self, from_size: int, to_size: int, initialiser=init_xavier):
        self.weights = initialiser((from_size, to_size))
        self.vocab_size = from_size

    def forward(self, tensor: Tensor):
        assert (
            tensor.requires_grad is False
        ), "Cannot embed a differentiable tensor"

        if tensor.is_vector:
            result = tensor.xp.stack(
                [self.weights._data[idx] for idx in tensor._data]
            )
            result = to_tensor(
                result,
                is_vector=True,
            )
        else:
            result = to_tensor(self.weights[tensor._data], is_vector=False)

        result.args = (tensor, self.weights)

        def _embed_back_fn(grad: Tensor):
            xp = grad.xp
            coef = xp.zeros((tensor.shape[-1], self.vocab_size))
            indices = xp.arange(tensor.shape[-1])

            coef[indices, tensor._data] = 1
            coef = to_tensor(coef, requires_grad=False)
            return Einsum("aB,aC->BC")(coef, grad)

        result.back_fns = (nothing, _embed_back_fn)
        return result

    def _raise_exception(self, *args):
        """
        I haven't figured out how 2nd order derivatives work yet so we'll
        raise an exception for now
        """
        raise NotImplementedError(
            "2nd order derivatives for embedding are not yet implemented"
        )

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


class EmbeddingV2(Layer):
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

    def forward(self, tensor: Tensor):
        assert (
            tensor.requires_grad is False
        ), "Cannot embed a differentiable tensor"

        if tensor.is_vector:
            result = tensor.xp.stack(
                [self.weights._data[idx] for idx in tensor._data]
            )
            result = to_tensor(
                result,
                is_vector=True,
            )
        else:
            result = to_tensor(self.weights[tensor._data], is_vector=False)

        result.args = (tensor, self.weights)

        def _embed_back_fn(grad: Tensor):
            xp = grad.xp
            out = xp.zeros(self.weights.shape)

            if grad.is_vector:
                for batch_idx, tokens in enumerate(tensor):
                    for token_idx, token in enumerate(tokens):
                        out[int(token._data)] += grad[batch_idx][
                            token_idx
                        ]._data
            else:
                for token_idx, token in enumerate(tensor):
                    out[int(token._data)] += grad[token_idx]._data
            return to_tensor(out)

        result.back_fns = (nothing, _embed_back_fn)
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
