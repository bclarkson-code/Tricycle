from functools import partial
import numpy as np

from llm_from_scratch.ops import (_no_grad, add, div, mul, negate, nothing,
                                  pow, reduce_sum, sub, tensor)


def partial_func_equals(a, b):
    assert a.func == b.func
    assert a.args == b.args


def grad_fn_equal(tensor_, ops):
    assert len(tensor_.grad_fn) == len(ops)

    for fn, op in zip(tensor_.grad_fn, ops):
        assert fn.func == _no_grad(op).func
        assert fn.args == _no_grad(op).args


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

    assert (x.grad == [1.0, 1.0, 1.0]).all()
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
