"""
Several layers can be grouped together into a single layer called a block
"""

from typing import Callable

from tricycle.activation import GeLU
from tricycle.initialisers import init_xavier
from tricycle.layers import Dense, Dropout, Layer
from tricycle.optimisers import Optimiser
from tricycle.tensor import Tensor


class MLPBlock(Layer):
    """
    A simple GPT-2 style MLP block with 2 linear layers around an activation
    function

    The size of the hidden dimension is expansion_ratio * the size of the
    input
    """

    embedding_dim: int
    dropout_prob: float
    expansion_ratio: float
    activation_fn: Callable
    linear_1: Dense
    linear_2: Dense
    dropout: Dropout

    def __init__(
        self,
        embedding_dim: int,
        dropout_prob: float,
        expansion_ratio: float = 4,
        activation_fn: Callable = GeLU(),
    ):
        self.linear_1 = Dense(
            from_size=embedding_dim,
            to_size=int(expansion_ratio * embedding_dim),
            initialiser=init_xavier,
        )
        self.linear_2 = Dense(
            from_size=int(expansion_ratio * embedding_dim),
            to_size=embedding_dim,
            initialiser=init_xavier,
        )
        self.dropout = Dropout(dropout_prob)
        self.activation_fn = activation_fn

    def forward(self, x: Tensor):
        x = self.linear_1(x)
        x = self.activation_fn(x)
        x = self.linear_2(x)
        x = self.dropout(x)
        return x

    def update(self, optimiser: Optimiser):
        self.linear_1.update(optimiser)
        self.linear_2.update(optimiser)

    def zero_grad(self):
        self.linear_1.zero_grad()
        self.linear_2.zero_grad()
