import numpy as np

from tricycle.einsum import Einsum
from tricycle.tensor import to_tensor


def test_vector_reduce():
    x = to_tensor(np.arange(5))
    op = Einsum("a->")
    result = op(x)
    assert result.close_to(10)

    result.backward()
    assert x.grad is not None
    assert x.grad.close_to(np.ones_like(x))


def test_matrix_reduce():
    x = to_tensor(np.arange(20).reshape(4, 5))
    op = Einsum("ab->")
    assert op(x) == 190

    op(x).backward()
    assert x.grad is not None
    assert x.grad.close_to(np.ones_like(x))


def test_matrix_partial_reduce():
    x = to_tensor(np.arange(20).reshape(4, 5))
    op = Einsum("ab->b")
    assert op(x).close_to([30, 34, 38, 42, 46])

    op(x).backward()
    assert x.grad is not None
    assert x.grad.close_to(np.ones_like(x))


def test_transpose():
    x = to_tensor(np.arange(20).reshape(4, 5))
    op = Einsum("ij->ji")
    assert op(x).close_to(x.T)

    op(x).backward()
    assert x.grad is not None
    assert x.grad.close_to(np.ones_like(x))
