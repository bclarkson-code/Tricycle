import numpy as np

from tricycle import GPU_ENABLED
from tricycle.activation import GLU, CudaGeLU, CudaReLU, GeLU, ReLU, Swish
from tricycle.tensor import Tensor


def test_relu():
    x = Tensor([-1, 0, 1])
    relu = ReLU()
    y = relu(x)
    assert y.close_to([0, 0, 1])


def test_swish():
    x = Tensor([-1, 0, 1])
    swish = Swish()
    y = swish(x)
    assert y.close_to([-0.26894142, 0.0, 0.73105858], rtol=1e-3)


def test_gelu_full():
    x = Tensor([-1, 0, 1])
    gelu = GeLU(approximate=False)
    y = gelu(x)
    assert y.close_to([-0.158808, 0.0, 0.841192], rtol=1e-3)


def test_gelu_batched():
    x = Tensor([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    x = x.to_batched()
    gelu = GeLU(approximate=False)
    y = gelu(x)
    assert y.close_to(
        [
            [-0.158808, 0.0, 0.841192],
            [-0.158808, 0.0, 0.841192],
            [-0.158808, 0.0, 0.841192],
        ],
        rtol=1e-3,
    )


def test_gelu_approx():
    x = Tensor([-1, 0, 1])
    gelu = GeLU(approximate=True)
    y = gelu(x)

    assert y.close_to(GeLU(approximate=False)(x), rtol=1e-3)


def test_glu():
    x = Tensor([-1, 0, 2])
    glu = GLU(size=3)
    glu.linear.weights = Tensor(np.ones(glu.linear.weights.shape))

    y = glu(x)
    assert y.close_to([0.73105858, 0.73105858, 0.73105858], rtol=1e-3)


if GPU_ENABLED:

    def test_relus_match():
        np.random.seed(0)
        INPUT_SHAPE = (32, int(2**15))
        random_data = np.random.random(INPUT_SHAPE) * 2 - 1
        tensor_1 = Tensor(random_data.copy())
        tensor_2 = Tensor(random_data.copy())

        tensor_1.to_gpu()
        tensor_2.to_gpu()

        relu = ReLU()
        output_1 = relu(tensor_1)
        output_1.backward()

        cuda_relu = CudaReLU()
        output_2 = cuda_relu(tensor_2)
        output_2.backward()

        assert output_1.close_to(output_2)
        assert tensor_1.grad.close_to(tensor_2.grad)

    def test_gelus_match():
        np.random.seed(0)
        INPUT_SHAPE = (32, int(2**15))
        random_data = np.random.random(INPUT_SHAPE) * 2 - 1
        tensor_1 = Tensor(random_data.copy())
        tensor_2 = Tensor(random_data.copy())

        tensor_1.to_gpu()
        tensor_2.to_gpu()

        relu = GeLU()
        output_1 = relu(tensor_1)
        output_1.backward()

        cuda_relu = CudaGeLU()
        output_2 = cuda_relu(tensor_2)
        output_2.backward()

        assert output_1.close_to(output_2)
        assert tensor_1.grad.close_to(tensor_2.grad, rtol=1e-5, atol=1e-6)
