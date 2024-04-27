import humanize
import numpy as np

from tricycle.blocks import GPT2TransformerBlock, GPT2TransformerBlockV4
from tricycle.configs import GPTConfig
from tricycle.layers import (
    Dense,
    DenseV3,
    Dropout,
    DropoutV7,
    Embedding,
    EmbeddingV2,
    Layer,
)
from tricycle.optimisers import Optimiser
from tricycle.tensor import Tensor, to_tensor


class GPT(Layer):
    def __init__(self, config: GPTConfig):
        self.embedding_dim = config.embedding_dim
        self.context_window = config.context_window
        self.token_embedding = Embedding(
            to_size=self.embedding_dim, from_size=config.vocab_size
        )
        self.position_embedding = Embedding(
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
        self.layers = [
            self.token_embedding,
            self.position_embedding,
            self.input_dropout,
            *self.blocks,
            self.head,
        ]

    def forward(self, tensor: Tensor):
        """
        Forward pass of the transformer. inputs is expected to be a one-hot
        encoded tensor
        """
        xp = tensor.xp
        if tensor.ndim == 1:
            n_batches = 1
            context_window = tensor.shape[-1]
            tensor._data = xp.expand_dims(tensor._data, 0)
        else:
            n_batches, context_window = tensor.shape
        assert n_batches <= self.context_window, (
            "Can't have more tokens than context window. ",
            f"Found {n_batches=} and {self.context_window=}",
        )

        position = to_tensor(
            xp.arange(context_window), requires_grad=False, dtype=int
        )

        pos_embedding = self.position_embedding(position)
        token_embedding = self.token_embedding(tensor)

        embedding = token_embedding + pos_embedding
        embedding = self.input_dropout(embedding)

        for block in self.blocks:
            embedding = block(embedding)

        return self.head(embedding)

    def zero_grad(self):
        self.token_embedding.zero_grad()
        self.position_embedding.zero_grad()
        self.head.zero_grad()
        for block in self.blocks:
            block.zero_grad()

    def update(self, optimiser: Optimiser):
        self.token_embedding.update(optimiser)
        self.position_embedding.update(optimiser)
        self.head.update(optimiser)
        for block in self.blocks:
            block.update(optimiser)

    def to_gpu(self, device: int = 0):
        self.token_embedding.to_gpu(device)
        self.position_embedding.to_gpu(device)
        for block in self.blocks:
            block.to_gpu(device)
        self.head.to_gpu(device)

    def from_gpu(self):
        self.token_embedding.from_gpu()
        self.position_embedding.from_gpu()
        for block in self.blocks:
            block.from_gpu()
        self.head.from_gpu()

    def display(self):
        stack = [(self, 0)]

        contents = []
        while stack:
            node, indent = stack.pop()

            tensors = list(node.tensors.values())
            shapes = [t.shape for t in tensors]
            size = sum(np.product(shape) for shape in shapes)
            contents.append((node.__class__.__name__, size, indent))

            stack.extend((layer, indent + 1) for layer in node.layers[::-1])

        total = 0
        for layer, size, n_indent in contents:
            total += size
            size = humanize.scientific(size) if size else ""
            indent = "  " * n_indent

            print(f"{indent}{layer}({size})")

        PARAM_SIZE = self.head.weights[0][0].dtype.itemsize
        total *= PARAM_SIZE

        print("\nTotal size:")
        print(f"  - {humanize.naturalsize(total)}")
        print("Total parameters:")
        print(f"  - {humanize.intword(total/PARAM_SIZE)}")


class GPTV2(Layer):
    def __init__(self, config: GPTConfig):
        self.embedding_dim = config.embedding_dim
        self.context_window = config.context_window
        self.token_embedding = EmbeddingV2(
            to_size=self.embedding_dim,
            from_size=config.vocab_size,
            name="token_embedding",
        )
        self.position_embedding = EmbeddingV2(
            to_size=self.embedding_dim,
            from_size=self.context_window,
            name="position_embedding",
        )
        self.input_dropout = DropoutV7(config.input_dropout_prob)

        self.blocks = [
            GPT2TransformerBlockV4(
                embedding_dim=self.embedding_dim,
                n_heads=config.n_heads,
                context_window=self.context_window,
                expansion_ratio=config.expansion_ratio,
                activation_fn=config.activation_fn,
                attention_dropout_prob=config.attention_dropout_prob,
            )
            for _ in range(config.n_layers)
        ]

        self.head = DenseV3(
            to_size=config.vocab_size,
            from_size=self.embedding_dim,
            name="head",
        )
        self.layers = [
            self.token_embedding,
            self.position_embedding,
            self.input_dropout,
            *self.blocks,
            self.head,
        ]

    def forward(self, tensor: Tensor):
        """
        Forward pass of the transformer. inputs is expected to be a one-hot
        encoded tensor
        """
        xp = tensor.xp
        if tensor.ndim == 1:
            n_tokens = 1
            context_window = tensor.shape[-1]
            tensor._data = xp.expand_dims(tensor._data, 0)
        else:
            n_tokens, context_window = tensor.shape
        assert n_tokens <= self.context_window, (
            "Can't have more tokens than context window. ",
            f"Found {n_tokens=} and {self.context_window=}",
        )

        position = to_tensor(
            xp.arange(context_window), requires_grad=False, dtype=int
        )

        pos_embedding = self.position_embedding(position)
        token_embedding = self.token_embedding(tensor)

        embedding = token_embedding + pos_embedding
        embedding = self.input_dropout(embedding)

        for block in self.blocks:
            embedding = block(embedding)

        return self.head(embedding)

    def zero_grad(self):
        self.token_embedding.zero_grad()
        self.position_embedding.zero_grad()
        self.head.zero_grad()
        for block in self.blocks:
            block.zero_grad()

    def update(self, optimiser: Optimiser):
        self.token_embedding.update(optimiser)
        self.position_embedding.update(optimiser)
        self.head.update(optimiser)
        for block in self.blocks:
            block.update(optimiser)

    def to_gpu(self, device: int = 0):
        self.token_embedding.to_gpu(device)
        self.position_embedding.to_gpu(device)
        for block in self.blocks:
            block.to_gpu(device)
        self.head.to_gpu(device)

    def from_gpu(self):
        self.token_embedding.from_gpu()
        self.position_embedding.from_gpu()
        for block in self.blocks:
            block.from_gpu()
        self.head.from_gpu()

    def display(self):
        stack = [(self, 0)]

        contents = []
        while stack:
            node, indent = stack.pop()

            tensors = list(node.tensors.values())
            shapes = [t.shape for t in tensors]
            size = sum(np.product(shape) for shape in shapes)
            contents.append((node.__class__.__name__, size, indent))

            stack.extend((layer, indent + 1) for layer in node.layers[::-1])

        total = 0
        for layer, size, n_indent in contents:
            total += size
            size = humanize.scientific(size) if size else ""
            indent = "  " * n_indent

            print(f"{indent}{layer}({size})")

        PARAM_SIZE = self.head.weights[0][0].dtype.itemsize
        total *= PARAM_SIZE

        print("\nTotal size:")
        print(f"  - {humanize.naturalsize(total)}")
        print("Total parameters:")
        print(f"  - {humanize.intword(total/PARAM_SIZE)}")
