"""
Several layers can be grouped together into a single layer called a block
"""

from tricycle.activation import GeLU
from tricycle.initialisers import init_xavier
from tricycle.layers import Dense, Layer
from tricycle.tensor import Tensor


class MLPBlock(Layer):
    """
    A simple GPT-2 style MLP block with 2 linear layers around an activation
    function
    """

    def __init__(
        self,
        embedding_dim: int,
        dropout: float,
        expansion_ratio: float = 4,
        activation_fn: callable = GeLU(),
    ):
        self.linear_1 = Dense(
            from_size=embedding_dim,
            to_size=int(expansion_ratio * embedding_dim),
            initialiser=init_xavier,
        )
        self.linear2 = Dense(
            from_size=int(expansion_ratio * embedding_dim),
            to_size=embedding_dim,
            initialiser=init_xavier,
        )
        self.dropout = dropout
        self.activation_fn = activation_fn

    def forward(self, x: Tensor):
        x = self.linear_1(x)
        x = self.activation_fn(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x
