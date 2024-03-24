import numpy as np

from tricycle.activation import GLU, GeLU, ReLU, SwiGLU, Swish
from tricycle.tensor import to_tensor


def test_relu():
    x = to_tensor([-1, 0, 1])
    relu = ReLU()
    y = relu(x)
    assert np.allclose(y, [0, 0, 1])


def test_swish():
    x = to_tensor([-1, 0, 1])
    swish = Swish()
    y = swish(x)
    assert y.close_to([-0.26894142, 0.0, 0.73105858])


def test_gelu_full():
    x = to_tensor([-1, 0, 1])
    gelu = GeLU(approximate=False)
    y = gelu(x)
    assert y.close_to([-0.15865525, 0.0, 0.84134475])


def test_gelu_approx():
    x = to_tensor([-1, 0, 1])
    gelu = GeLU(approximate=True)
    y = gelu(x)

    assert y.close_to(GeLU(approximate=False)(x), rtol=1e-3)


def test_glu():
    x = to_tensor([-1, 0, 2])
    glu = GLU(size=3)
    glu.linear.weights = to_tensor(np.ones_like(glu.linear.weights))

    y = glu(x)
    assert y.close_to([0.73105858, 0.73105858, 0.73105858])


def test_swiglu():
    x = to_tensor([-1, 0, 1, 2])
    swiglu = SwiGLU(size=4)
    swiglu.linear.weights = to_tensor(np.ones_like(swiglu.linear.weights))

    y = swiglu(x)
    assert y.close_to([3.52318831, 3.52318831, 3.52318831, 3.52318831])
