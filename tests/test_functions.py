from tricycle.functions import Sigmoid
from tricycle.tensor import to_tensor


def test_sigmoid():
    in_tensor = to_tensor([0, 1, 2, 3])
    out_tensor = Sigmoid()(in_tensor)

    assert out_tensor.shape == (4,)
    assert out_tensor.close_to(
        [0.5, 0.73105858, 0.88079708, 0.95257413], rtol=1e-3
    )

    out_tensor.backward()
    correct_grad = out_tensor * (1 - out_tensor)

    assert in_tensor.grad is not None
    assert in_tensor.grad.close_to(correct_grad, rtol=1e-3, atol=1e-3)
