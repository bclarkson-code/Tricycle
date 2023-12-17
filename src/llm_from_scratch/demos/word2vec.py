from pathlib import Path

import mlflow
import numpy as np
import torch
from omegaconf import OmegaConf
from tiktoken import get_encoding
from torch.nn import Linear, Module, ReLU, Sequential
from tqdm import tqdm


class WindowDataset(torch.utils.data.Dataset):
    """
    Return the tokens on the left and right of a token as the input
    along with the token itself as the label
    """

    window_size: int

    def __init__(
        self,
        text: str,
        window_size: int = 5,
        encoding: str = "p50k_base",
        save_path: str = "bee_movie_tokens.npy",
    ):
        self.window_size = window_size
        self.encoding = get_encoding(encoding)
        self.save_path = Path(save_path)
        self.tokens = self.load_tokens(text)

    def load_tokens(self, text: str):
        if self.save_path.exists():
            return np.load(self.save_path)

        tokens = self.encoding.encode(text)
        np.save(self.save_path, tokens)
        return tokens

    def __getitem__(self, idx: int) -> tuple[np.ndarray, int]:
        left_idx = max(0, idx - self.window_size)
        right_idx = min(len(self.tokens), idx + self.window_size)
        left = self.tokens[left_idx:idx]
        right = self.tokens[idx + 1 : right_idx]
        input_indices = np.concatenate([left, right]).astype(int)
        inputs = self.to_binary_vector(input_indices)

        label = self.tokens[idx]

        return inputs, label

    def to_binary_vector(self, indices: np.ndarray) -> np.ndarray:
        zeros = np.zeros(self.encoding.n_vocab, dtype=np.float32)
        zeros[indices] = 1
        return zeros

    def __len__(self) -> int:
        return len(self.tokens)


class Model(Module):
    """
    A simple feed forward model that is trained with Word2Vec (continuous
    bag of words). The goal of the model is to produce an embedding vector
    for a given word.
    """

    vocab_size: int
    encoder: Sequential
    decoder: Linear
    embedding_size: int

    def __init__(
        self, vocab_size: int, embedding_size: int = 256, weight_init="xavier_norm"
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.weight_init = weight_init
        self.encoder = Sequential(
            Linear(self.vocab_size, self.embedding_size),
            ReLU(),
        )
        self.decoder = Sequential(Linear(self.embedding_size, self.vocab_size))

        self.encoder.apply(self.initialise_weights)
        self.decoder.apply(self.initialise_weights)

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def initialise_weights(self, layer: Module) -> None:
        if not isinstance(layer, Linear):
            return

        match self.weight_init:
            case "xavier_norm":
                weight_init_fn = torch.nn.init.xavier_normal_
            case _:
                raise NotImplementedError(f"Unknown weight init {self.weight_init}")

        weight_init_fn(layer.weight)


def load_optimiser(config: OmegaConf, model: Module) -> None:
    match config.optimiser.type:
        case "SGD":
            optimiser = torch.optim.SGD
        case _:
            raise NotImplementedError(f"Unknown optimiser {config.optimiser.type}")

    optimiser_config = config.optimiser
    del optimiser_config.type
    return optimiser(model.parameters(), **optimiser_config)


def load_loss_fn(config: OmegaConf) -> Module:
    match config.loss.type:
        case "cross_entropy":
            loss_fn = torch.nn.CrossEntropyLoss
        case _:
            raise NotImplementedError(f"Unknown loss {config.loss.type}")

    loss_fn_config = config.loss
    del loss_fn_config.type
    return loss_fn(**loss_fn_config)


def init_logger(config: OmegaConf):
    mlflow.set_tracking_uri(config.logger.tracking_url)
    mlflow.set_experiment(config.logger.project)


def train(
    model: Module,
    train_dl: torch.utils.data.DataLoader,
    optimiser,
    loss_fn,
    config: OmegaConf,
    epoch: int,
):
    for idx, (inputs, labels) in tqdm(
        enumerate(train_dl), leave=False, desc="Training", total=len(train_dl)
    ):
        optimiser.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        mlflow.log_metric(
            "train_loss",
            loss.item() / config.dataset.batch_size,
            step=epoch * len(train_dl) + idx,
        )
        optimiser.step()


def test(
    model: Module,
    test_dl: torch.utils.data.DataLoader,
    loss_fn,
    config: OmegaConf,
    epoch: int,
    steps_per_epoch: int,
):
    losses = []
    with torch.no_grad():
        for inputs, labels in tqdm(test_dl, leave=False, desc="Testing"):
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            losses.append(loss.item() / config.dataset.batch_size)
    mlflow.log_metric("test_loss", np.mean(losses), step=(epoch + 1) * steps_per_epoch)


if __name__ == "__main__":
    config = OmegaConf.create(
        {
            "model": {"embedding_size": 256, "weight_init": "xavier_norm"},
            "dataset": {
                "window_size": 5,
                "encoding": "p50k_base",
                "train_fraction": 0.8,
                "batch_size": 32,
            },
            "optimiser": {"type": "SGD", "lr": 1e-3, "momentum": 0.9},
            "loss": {"type": "cross_entropy"},
            "logger": {
                "project": "llm_from_scratch/word2vec",
                "tracking_url": "http://localhost:8080",
            },
            "training": {"epochs": 10},
        }
    )
    with open("bee-movie.txt", "r") as f:
        text = f.read()

    test_start_idx = int(len(text) * config.dataset.train_fraction)
    train_text = text[:test_start_idx]
    test_text = text[test_start_idx:]

    train_ds = WindowDataset(train_text)
    test_ds = WindowDataset(test_text)

    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=config.dataset.batch_size, shuffle=True
    )
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=config.dataset.batch_size)

    model = Model(train_ds.encoding.n_vocab, **config.model)

    optimiser = load_optimiser(config, model)
    loss_fn = load_loss_fn(config)

    init_logger(config)
    steps_per_epoch = len(train_dl)

    with mlflow.start_run():
        for epoch in tqdm(range(config.training.epochs), leave=False, desc="Epochs"):
            train(model, train_dl, optimiser, loss_fn, config, epoch)
            test(model, test_dl, loss_fn, config, epoch, steps_per_epoch)
