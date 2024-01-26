import numpy as np

from tricycle.ops import softmax
from tricycle.reduce import radd
from tricycle.tensor import to_tensor


def test_softmax():
    in_tensor = to_tensor(np.arange(12).reshape(3, 4), name="in_tensor")

    out_tensor = softmax(in_tensor)

    assert out_tensor.shape == (3, 4)
    assert np.allclose(radd(out_tensor, "ij->i"), [1, 1, 1])
    assert np.allclose(
        out_tensor,
        np.array(
            [
                [0.0320586, 0.08714432, 0.23688282, 0.64391426],
                [0.0320586, 0.08714432, 0.23688282, 0.64391426],
                [0.0320586, 0.08714432, 0.23688282, 0.64391426],
            ]
        ),
    )

    out_tensor.backward()

    breakpoint()
    assert np.allclose(
        in_tensor.grad,
        np.ones_like(in_tensor),
    )
