import numpy as np

from tricycle.activation import GLU, GeLU, ReLU, SwiGLU, Swish
from tricycle.tensor import to_tensor


def test_relu():
    x = to_tensor([-1, 0, 1])
    relu = ReLU()
    y = relu(x)
    assert y.close_to([0, 0, 1])


def test_swish():
    x = to_tensor([-1, 0, 1])
    swish = Swish()
    y = swish(x)
    assert y.close_to([-0.26894142, 0.0, 0.73105858])


def test_gelu_full():
    x = to_tensor([-1, 0, 1])
    gelu = GeLU(approximate=False)
    y = gelu(x)
    assert y.close_to([-0.158808, 0.0, 0.841192])


def test_gelu_batched():
    x = to_tensor([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    x = x.to_batched()
    gelu = GeLU(approximate=False)
    y = gelu(x)
    assert y.close_to(
        [
            [-0.158808, 0.0, 0.841192],
            [-0.158808, 0.0, 0.841192],
            [-0.158808, 0.0, 0.841192],
        ]
    )


def test_gelu_approx():
    x = to_tensor([-1, 0, 1])
    gelu = GeLU(approximate=True)
    y = gelu(x)

    assert y.close_to(GeLU(approximate=False)(x), rtol=1e-3)


def test_glu():
    x = to_tensor([-1, 0, 2])
    glu = GLU(size=3)
    glu.linear.weights = to_tensor(np.ones(glu.linear.weights.shape))

    y = glu(x)
    assert y.close_to([0.73105858, 0.73105858, 0.73105858])


def test_swiglu():
    x = to_tensor([-1, 0, 1, 2])
    swiglu = SwiGLU(size=4)
    swiglu.linear.weights = to_tensor(np.ones(swiglu.linear.weights.shape))

    y = swiglu(x)
    assert y.close_to([3.52318831, 3.52318831, 3.52318831, 3.52318831])

    y.backward()

    assert swiglu.linear.weights.grad is not None
    weight_grad = swiglu.linear.weights.grad
    assert weight_grad.close_to(
        [
            [
                -1.76159416,
                -1.76159416,
                -1.76159416,
                -1.76159416,
                -2.1815685,
                -2.1815685,
                -2.1815685,
                -2.1815685,
            ],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [
                1.76159416,
                1.76159416,
                1.76159416,
                1.76159416,
                2.1815685,
                2.1815685,
                2.1815685,
                2.1815685,
            ],
            [
                3.52318831,
                3.52318831,
                3.52318831,
                3.52318831,
                4.363137,
                4.363137,
                4.363137,
                4.363137,
            ],
        ]
    )

    assert swiglu.bias.grad is not None
    assert swiglu.bias.grad.close_to(3.359794732912208)
