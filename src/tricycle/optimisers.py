import cupy as cp

from tricycle.tensor import Tensor, to_tensor


class Optimiser:
    def __call__(self, tensor: Tensor) -> Tensor:
        raise NotImplementedError

    def _reset_grad(self, tensor: Tensor):
        tensor.grad = None
        tensor.args = None
        tensor.back_fns = None
        return tensor


class StochasticGradientDescent(Optimiser):
    def __init__(
        self,
        learning_rate: float,
        weight_decay: float | None = None,
        momentum: float | None = None,
    ):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.momentum = momentum

        self.momentum_store = {}

    def update_weight(self, tensor: Tensor):
        """
        Perform a gradient update on a tensor, optionally
        including weight decay and momentum
        """
        assert tensor.grad is not None

        if tensor.grad.is_vector:
            tensor.grad = tensor.grad.from_vector().e("z...->...")

        grad = self.learning_rate * tensor.grad

        if self.weight_decay is not None:
            wd = self.learning_rate * self.weight_decay * tensor
            grad += to_tensor(wd, name=f"weight_decay({self.weight_decay})")

        if self.momentum is not None and self.momentum > 0:
            if tensor._id not in self.momentum_store:
                last_momentum = to_tensor(tensor.xp.zeros(grad.shape))
            else:
                last_momentum = self.momentum_store[tensor._id]

            grad += self.momentum * last_momentum
            self.momentum_store[tensor._id] = to_tensor(grad._data)

        # update the value only, leave everything else
        result = to_tensor(
            tensor - grad,
            requires_grad=tensor.requires_grad,
            name=tensor.name,
            is_vector=tensor.is_vector,
            _id=tensor._id,
        )

        assert result.shape == tensor.shape

        del tensor
        del grad

        return result

    def __call__(self, tensor: Tensor) -> Tensor:
        return self._reset_grad(self.update_weight(tensor))


class AdamW(Optimiser):
    def __init__(
        self,
        learning_rate=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    ):
        self.learning_rate = learning_rate
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        # we compute the updates dynamically so we'll need to remember to
        # call this
        self.t += 1

    def update_weight(self, tensor: Tensor) -> Tensor:
        key = tensor._id
        xp = tensor.xp

        # initialise stores
        if key not in self.m:
            self.m[key] = xp.zeros_like(tensor._data)
        if key not in self.v:
            self.v[key] = tensor.xp.zeros_like(tensor._data)

        self.m[key] = (
            self.betas[0] * self.m[key]
            + (1 - self.betas[0]) * tensor.grad._data
        )

        self.v[key] = self.betas[1] * self.v[key] + (1 - self.betas[1]) * (
            tensor.grad._data * tensor.grad._data
        )

        m_hat = self.m[key] / (1 - self.betas[0] ** self.t)
        v_hat = self.v[key] / (1 - self.betas[1] ** self.t)

        tensor._data -= self.learning_rate * (
            m_hat / (xp.sqrt(v_hat) + self.eps)
            + self.weight_decay * tensor._data
        )

        tensor.grad._data.fill(0)

        return tensor

    def __call__(self, tensor: Tensor) -> Tensor:
        return self._reset_grad(self.update_weight(tensor))


# @cp.fuse()
def _calculate_momentum(
    beta_0: float, prev_momentum: cp.ndarray, grad: cp.ndarray
) -> cp.ndarray:
    return beta_0 * prev_momentum + (1 - beta_0) * grad


# @cp.fuse()
def _calculate_v(
    beta_1: float, prev_v: cp.ndarray, grad: cp.ndarray
) -> cp.ndarray:
    return beta_1 * prev_v + (1 - beta_1) * (grad**2)


# @cp.fuse()
def _estimate_param(beta: float, val: cp.ndarray, t: float) -> cp.ndarray:
    return val / (1 - beta**t)


# @cp.fuse()
def _update(
    tensor: cp.ndarray,
    learning_rate: float,
    m_hat: cp.ndarray,
    v_hat: cp.ndarray,
    eps: float,
    weight_decay: float,
):
    """
    Inplace update tensor
    """
    coef = m_hat / (cp.sqrt(v_hat) + eps)
    return learning_rate * coef + weight_decay * tensor


class AdamWV2(Optimiser):
    def __init__(
        self,
        learning_rate=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    ):
        self.learning_rate = learning_rate
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        # we compute the updates dynamically so we'll need to remember to
        # call this
        self.t += 1

    def update_weight(self, tensor: Tensor) -> Tensor:
        key = tensor._id
        xp = tensor.xp

        # initialise stores
        if key not in self.m:
            self.m[key] = xp.zeros_like(tensor._data)
        if key not in self.v:
            self.v[key] = tensor.xp.zeros_like(tensor._data)

        self.m[key] = _calculate_momentum(
            self.betas[0], self.m[key], tensor.grad._data
        )

        self.v[key] = _calculate_v(
            self.betas[1], self.v[key], tensor.grad._data
        )

        m_hat = _estimate_param(self.betas[0], self.m[key], self.t)
        v_hat = _estimate_param(self.betas[1], self.v[key], self.t)

        # this update is don inplace to avoid synchronisation
        tensor._data -= _update(
            tensor._data,
            self.learning_rate,
            m_hat,
            v_hat,
            self.eps,
            self.weight_decay,
        )
        return tensor

    def __call__(self, tensor: Tensor) -> Tensor:
        tensor = self.update_weight(tensor)
        tensor = self._reset_grad(tensor)
        return tensor
