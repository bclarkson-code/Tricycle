import numpy as np

from llm_from_scratch.ops import mean, softmax, tensor, sigmoid


def test_can_mean():
    assert mean(tensor([1, 2, 3])) == 2


def test_can_softmax():
    result = softmax(tensor([0, 1, 0]))
    expected = tensor([0.21194156, 0.57611688, 0.21194156])
    assert np.allclose(result, expected)


def test_can_sigmoid():
    result = sigmoid(tensor([1, 2, 3]))
    expected = tensor([0.73105858, 0.88079708, 0.95257413])
    assert np.allclose(result, expected)
