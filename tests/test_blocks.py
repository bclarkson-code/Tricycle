import numpy as np

from tricycle.blocks import MLPBlock
from tricycle.tensor import to_tensor


def test_MLPBlock():
    np.random.seed(0)
    in_tensor = to_tensor(np.arange(12, dtype=float).reshape(3, 4))
    block = MLPBlock(embedding_dim=4, expansion_ratio=4, dropout_prob=0.5)

    assert block.linear_1.weights.shape == (4, 16)
    assert block.linear_2.weights.shape == (16, 4)

    block.linear_1.weights = to_tensor(np.ones_like(block.linear_1.weights))
    block.linear_2.weights = to_tensor(np.ones_like(block.linear_2.weights))

    out_tensor = block(in_tensor.to_vector())

    assert out_tensor.shape == (3, 4)

    correct_output = np.array(
        [[96, 0, 96, 0], [0, 0, 352, 352], [608, 608, 608, 0]]
    )
    correct_output = to_tensor(correct_output)

    assert out_tensor.is_vector
    assert out_tensor.close_to(correct_output)

    out_tensor.backward()

    assert in_tensor.grad is not None
    correct_grad = to_tensor(
        [
            [-44.59691787, -44.59691787, -44.59691787, -44.59691787],
            [-248.8553654, -248.8553654, -248.8553654, -248.8553654],
            [-679.67071945, -679.67071945, -679.67071945, -679.67071945],
        ]
    )

    assert in_tensor.grad.close_to(correct_grad)
