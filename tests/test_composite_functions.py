import numpy as np

from tricycle.ops import mean, sigmoid, softmax, tensor


def test_can_mean():
    assert mean(tensor([1, 2, 3])) == 2


def test_can_softmax():
    # Single row
    result = softmax(tensor([0, 1, 0]))
    expected = tensor([0.21194156, 0.57611688, 0.21194156])
    assert np.allclose(result, expected)

    # Multiple rows
    result = softmax(tensor([[0, 1, 0], [1, 0, 0]]))
    expected = tensor(
        [[0.21194156, 0.57611688, 0.21194156], [0.57611688, 0.21194156, 0.21194156]]
    )
    assert np.allclose(result, expected)

def test_can_differentiate_softmax():
    x = tensor([1, 2, 3])
    x.name = "x"
    z = softmax(x)
    z.name = "z"
    z.backward()

    assert np.allclose(x.grad, [-0.09003057, -0.24472847,  0.33475904])



def test_can_sigmoid():
    result = sigmoid(tensor([1, 2, 3]))
    expected = tensor([0.73105858, 0.88079708, 0.95257413])
    assert np.allclose(result, expected)
