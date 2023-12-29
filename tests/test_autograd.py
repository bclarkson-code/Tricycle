from functools import partial

import numpy as np

from llm_from_scratch.ops import (add, cos, div, einsum, exp, log, matmul, max,
                                  min, mul, negate, no_grad, nothing, pow,
                                  reduce_sum, sin, sqrt, sub, tensor)


def partial_func_equals(a, b):
    assert a.func == b.func
    assert a.args == b.args


def grad_fn_equal(tensor_, ops):
    assert len(tensor_.grad_fn) == len(ops)

    for fn, op in zip(tensor_.grad_fn, ops):
        if isinstance(op, partial):
            partial_func_equals(fn, op)
        else:
            assert fn == op


def test_can_differentiate_subtract():
    x = tensor(1.0)
    y = tensor(2.0)

    z = sub(y, x)

    assert z == 1.0

    z.backward()

    assert x.grad == -1.0
    assert y.grad == 1.0
    grad_fn_equal(x, [negate])
    grad_fn_equal(y, [nothing])


def test_can_differentiate_negate():
    x = tensor(1.0)

    z = negate(x)

    assert z == -1.0

    z.backward()

    assert x.grad == -1.0
    grad_fn_equal(x, [negate])


def test_can_differentiate_nothing():
    x = tensor(1.0)

    z = nothing(x)

    assert z == 1.0

    z.backward()

    assert x.grad == 1.0
    grad_fn_equal(x, [nothing])


def test_can_differentiate_add():
    x = tensor(1.0)
    y = tensor(2.0)

    z = add(x, y)

    assert z == 3.0

    z.backward()

    assert x.grad == 1.0
    grad_fn_equal(x, [nothing])

    assert y.grad == 1.0
    grad_fn_equal(y, [nothing])


def test_can_differentiate_multiply():
    x = tensor(1.0)
    y = tensor(2.0)

    z = mul(x, y)

    assert z == 2.0

    z.backward()

    assert x.grad == 2.0
    grad_fn_equal(x, [partial(mul, y=y)])

    assert y.grad == 1.0
    grad_fn_equal(x, [partial(mul, y=x)])


def test_can_differentiate_divide():
    x = tensor(1.0)
    y = tensor(2.0)

    z = div(x, y)

    assert z == 0.5

    z.backward()

    assert x.grad == 0.5
    grad_fn_equal(x, [partial(div, y=x)])

    assert y.grad == -0.25
    assert len(y.grad_fn) == 1
    grad_fn = y.grad_fn[0]
    assert grad_fn.__name__ == "diff_div"


def test_can_differentiate_reduce_sum():
    x = tensor([1.0, 2.0, 3.0])
    z = reduce_sum(x)
    assert z == 6.0

    z.backward()

    assert x.grad == 3.0
    grad_fn_equal(x, [reduce_sum])


def test_can_differentiate_power():
    x = tensor(2.0)
    y = tensor(3.0)

    z = pow(x, y)

    assert z == 8.0

    z.backward()

    assert x.grad == 12.0
    assert len(x.grad_fn) == 1
    grad_fn = x.grad_fn[0]
    assert grad_fn.__name__ == "diff_power_arg_1"

    assert y.grad == 8.0 * np.log(2.0)
    assert len(y.grad_fn) == 1
    grad_fn = y.grad_fn[0]
    assert grad_fn.__name__ == "diff_power_arg_2"


def test_can_differentiate_exp():
    x = tensor(2.0)

    z = exp(x)

    assert z == np.exp(2.0)

    z.backward()

    assert x.grad == np.exp(1.0)
    grad_fn_equal(x, [exp])


def test_can_differentiate_log():
    x = tensor(2.0)

    z = log(x)

    assert z == np.log(2.0)

    z.backward()

    assert x.grad == 1.0
    assert len(x.grad_fn) == 1


def test_can_differentiate_sqrt():
    x = tensor(2.0)

    z = sqrt(x)

    assert z == np.sqrt(2.0)

    z.backward()

    assert x.grad == 0.5
    assert len(x.grad_fn) == 1
    grad_fn = x.grad_fn[0]
    assert grad_fn.__name__ == "diff_sqrt"


def test_can_differentiate_sin():
    x = tensor(2.0)

    z = sin(x)

    assert z == np.sin(2.0)

    z.backward()

    assert x.grad == np.cos(1.0)
    grad_fn_equal(x, [cos])


def test_can_differentiate_cos():
    x = tensor(2.0)

    z = cos(x)

    assert z == np.cos(2.0)

    z.backward()

    assert x.grad == -np.sin(1.0)
    assert len(x.grad_fn) == 1
    grad_fn = x.grad_fn[0]
    assert grad_fn.__name__ == "diff_cos"


def test_can_differentiate_max():
    x = tensor([1.0, 2.0, 3.0])

    z = max(x)

    assert z == 3.0
    z.backward()

    assert len(x.grad_fn) == 1
    grad_fn = x.grad_fn[0]
    assert grad_fn.__name__ == "diff_max"


def test_can_differentiate_min():
    x = tensor([1.0, 2.0, 3.0])

    z = min(x)

    assert z == 1.0
    z.backward()

    assert len(x.grad_fn) == 1
    grad_fn = x.grad_fn[0]
    assert grad_fn.__name__ == "diff_min"


def test_can_differentiate_matmul():
    a = tensor([[1.0, 2.0], [3.0, 4.0]])
    b = tensor([[5.0, 6.0], [7.0, 8.0]])

    z = matmul(a, b)

    assert np.allclose(z, tensor([[19.0, 22.0], [43.0, 50.0]])), z

    z.backward()

    assert np.allclose(np.array(a.grad), np.array([[12.0, 14.0], [12.0, 14.0]]))

    assert np.allclose(np.array(b.grad), np.array([[3.0, 3.0], [7.0, 7.0]]))


def test_can_differentiate_einsum():
    a = np.arange(12).reshape(3, 4)
    b = np.arange(12).reshape(4, 3)

    a = tensor(a)
    b = tensor(b)

    output = np.array([[42, 48, 54], [114, 136, 158], [186, 224, 262]])

    z = einsum(a, b, subscripts="ij,jk->ik")
    assert np.allclose(z, output)

    z.backward()

    assert np.allclose(a.grad, np.array([[18, 22, 26], [18, 22, 26], [18, 22, 26]]))
    assert np.allclose(
        b.grad, np.array([[6, 6, 6], [22, 22, 22], [38, 38, 38]])
    ), b.grad


def test_can_combine_ops_correctly():
    x = tensor([1.0, 2.0, 3.0])
    y = tensor([4.0, 5.0, 6.0])

    z = ((x + y) * 0.3) ** 2

    assert np.allclose(z, tensor([2.25, 4.41, 7.29]))

    z.backward()

    assert np.allclose(np.array(x.grad), 0.18 * np.array(x + y))
    assert np.allclose(np.array(y.grad), 0.18 * np.array(x + y))


def test_can_do_linear_regression_backprop():
    x = tensor(np.linspace(-1.0, 1.0, 100))

    # hard code slope = 2, intercept = -1
    y = tensor(np.linspace(-1.0, 1.0, 100) * 2 - 1)

    slope = tensor(0.01)  # start off with a small slope
    intercept = tensor(0.0)  # start off with a small intercept

    learning_rate = tensor(0.1)

    prev_loss = np.inf
    for _ in range(100):
        z = slope * x + intercept

        # Calculate mean squared error
        loss = reduce_sum(((z - y) ** 2) / 100)

        # Make sure loss is decreasing
        assert prev_loss > loss
        prev_loss = loss

        # Do backprop
        loss.backward()

        with no_grad():
            slope = slope - slope.grad * learning_rate
            intercept = intercept - intercept.grad * learning_rate

    # Make sure we are close to the true slope and intercept
    assert np.allclose(slope, 2.0, rtol=0.1)
    assert np.allclose(intercept, -1.0, rtol=0.1)
