import numpy as np

from tricycle.blocks import GPT2TransformerBlock
from tricycle.configs import GPTConfig
from tricycle.layers import Dense, Dropout, Layer
from tricycle.tensor import Tensor, to_tensor


class PicoGPT(Layer):
    def __init__(self, config: GPTConfig):
        self.embedding_dim = config.embedding_dim
        self.context_window = config.context_window
        self.token_embedding = Dense(
            to_size=self.embedding_dim, from_size=config.vocab_size
        )
        self.position_embedding = Dense(
            to_size=self.embedding_dim, from_size=self.context_window
        )
        self.input_dropout = Dropout(config.input_dropout_prob)

        self.blocks = [
            GPT2TransformerBlock(
                embedding_dim=self.embedding_dim,
                n_heads=config.n_heads,
                context_window=self.context_window,
                expansion_ratio=config.expansion_ratio,
                activation_fn=config.activation_fn,
                attention_dropout_prob=config.attention_dropout_prob,
            )
            for _ in range(config.n_layers)
        ]

        self.head = Dense(
            to_size=config.vocab_size, from_size=self.embedding_dim
        )

    def forward(self, x: Tensor):
        """
        Forward pass of the transformer. inputs is expected to be a one-hot
        encoded tensor
        """
        _, n_tokens, _ = x.shape
        assert n_tokens <= self.context_window, (
            "Can't have more tokens than context window. ",
            f"Found {n_tokens=} and {self.context_window=}",
        )

        position = to_tensor(np.arange(n_tokens))

        pos_embedding = self.position_embedding(position)
        token_embedding = self.token_embedding(x)

        embedding = token_embedding + pos_embedding
        embedding = self.input_dropout(embedding)

        for block in self.blocks:
            embedding = block(embedding)

        return self.head(embedding)
