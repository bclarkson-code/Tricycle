from abc import ABC, abstractmethod
from string import ascii_letters
from typing import Sequence

from tricycle.binary import bmask, bmul
from tricycle.einsum import Einsum
from tricycle.initialisers import init_xavier
from tricycle.optimisers import Optimiser
from tricycle.tensor import Tensor, nothing, to_tensor


class Layer(ABC):
    tensors: dict[str, Tensor] = {}
    layers: Sequence["Layer"] = []

    @abstractmethod
    def forward(self, tensor: Tensor):
        raise NotImplementedError

    def __call__(self, tensor: Tensor):
        return self.forward(tensor)

    def update(self, optimiser: Optimiser):
        pass

    def zero_grad(self):
        pass

    def to_gpu(self):
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

    def to_gpu(self):
        self.weights.to_gpu()
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

    def to_gpu(self):
        self.weights.to_gpu()
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
        self.from_size = from_size
        self.to_size = to_size
        self.tensors = {"weights": self.weights}

    def _einsum_fn(self, subscript, tensor):
        def back_einsum(grad):
            result = tensor.xp.einsum(subscript, tensor._data, grad._data)
            return to_tensor(
                result,
                requires_grad=tensor.requires_grad,
                name="back_dense",
            )

        return back_einsum

    def forward(self, tensor: Tensor):
        if tensor.ndim == 1:
            result = self._forward(tensor, "a,aB->B", "a,B->aB", "aB,B->a")
        elif tensor.ndim == 2:
            result = self._forward(
                tensor, "Ab,bC->AC", "Ab,AC->bC", "bC,AC->Ab"
            )
        elif tensor.ndim == 3 and tensor.is_vector:
            result = self._forward(
                tensor, "zAb,bC->zAC", "zAb,zAC->bC", "bC,zAC->Ab"
            )
        else:
            raise NotImplementedError(
                f"Cannot pass tensor with shape {tensor.shape} "
                f"and {tensor.is_vector=}"
                "through a Dense layer"
            )

        result.name = "dense"
        return result

    def _forward(self, tensor, subscript, weight_subscript, tensor_subscript):
        result = to_tensor(
            tensor.xp.einsum(subscript, tensor._data, self.weights._data)
        )
        weight_back_fn = self._einsum_fn(weight_subscript, tensor)
        grad_back_fn = self._einsum_fn(tensor_subscript, self.weights)
        result.args = (self.weights, tensor)
        result.back_fns = (weight_back_fn, grad_back_fn)
        result.is_vector = tensor.is_vector

        return result

    def update(self, optimiser: Optimiser):
        self.weights = optimiser(self.weights)

    def zero_grad(self):
        self.weights.grad = None

    def to_gpu(self):
        self.weights.to_gpu()
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

    def to_gpu(self):
        self.weights.to_gpu()
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


class LayerNorm(Layer):
    """
    Normalise each tensor individually
    """

    def forward(self, tensor: Tensor):
        return tensor.normalise()


class Embedding(Layer):
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

    def to_gpu(self):
        self.weights.to_gpu()
        return self

    def from_gpu(self):
        self.weights.from_gpu()
        return self


class EmbeddingV2(Layer):
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
            if grad.is_vector:
                out = xp.zeros((grad.shape[0], *self.weights.shape))
            else:
                out = xp.zeros(self.weights.shape)

            if grad.is_vector:
                for batch_idx, tokens in enumerate(tensor):
                    for token_idx, token in enumerate(tokens):
                        out[batch_idx][int(token._data)] += grad[batch_idx][
                            token_idx
                        ]._data
            else:
                for token_idx, token in enumerate(tensor):
                    out[int(token._data)] += grad[token_idx]._data
            return to_tensor(out)

        result.back_fns = (nothing, _embed_back_fn)
        return result


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

    def to_gpu(self):
        for layer in self.layers:
            layer.to_gpu()

    def from_gpu(self):
        for layer in self.layers:
            layer.from_gpu()
