import numpy as np

from tricycle.einsum import Einsum, Subscript
from tricycle.tensor import to_tensor


def test_batched_reduce():
    x = to_tensor(np.arange(5))
    op = Einsum("a->")
    result = op(x)
    assert result.close_to(10)

    result.backward()
    assert x.grad is not None
    assert x.grad.close_to(np.ones(x.shape))


def test_matrix_reduce():
    x = to_tensor(np.arange(20).reshape(4, 5))
    op = Einsum("ab->")
    assert op(x) == 190

    op(x).backward()
    assert x.grad is not None
    assert x.grad.close_to(np.ones(x.shape))


def test_matrix_partial_reduce():
    x = to_tensor(np.arange(20).reshape(4, 5))
    op = Einsum("ab->b")
    assert op(x).close_to([30, 34, 38, 42, 46])

    op(x).backward()
    assert x.grad is not None
    assert x.grad.close_to(np.ones(x.shape))


def test_transpose():
    x = to_tensor(np.arange(20).reshape(4, 5))
    op = Einsum("ij->ji")
    assert op(x).close_to(x.array.T)

    op(x).backward()
    assert x.grad is not None
    assert x.grad.close_to(np.ones(x.shape))


def test_parse_subscripts():
    subscript = Subscript("a,b->ab")
    assert subscript.inputs == [["a"], ["b"]]
    assert subscript.output == ["a", "b"]

    subscript = Subscript("a,b->")
    assert subscript.inputs == [["a"], ["b"]]
    assert subscript.output == []

    subscript = Subscript("...a,b...->ab...")
    assert subscript.inputs == [["...", "a"], ["b", "..."]]
    assert subscript.output == ["a", "b", "..."]

    subscript = Subscript("...,...->...")
    assert subscript.inputs == [["..."], ["..."]]
    assert subscript.output == ["..."]

    subscript = Subscript("z...,z...->z...")
    assert subscript.inputs == [["z", "..."], ["z", "..."]]
    assert subscript.output == ["z", "..."]


def test_can_parse_split():
    inputs = [["a"], ["b"]]
    output = ["a", "b"]
    assert Subscript.join(inputs, output) == "a,b->ab"

    inputs = [["a"], ["b"]]
    output = []
    assert Subscript.join(inputs, output) == "a,b->"

    inputs = [["...", "a"], ["b", "..."]]
    output = ["a", "b", "..."]
    assert Subscript.join(inputs, output) == "...a,b...->ab..."

    inputs = [["..."], ["..."]]
    output = ["..."]
    assert Subscript.join(inputs, output) == "...,...->..."

    inputs = [["z", "..."], ["z", "..."]]
    output = ["z", "..."]
    assert Subscript.join(inputs, output) == "z...,z...->z..."
