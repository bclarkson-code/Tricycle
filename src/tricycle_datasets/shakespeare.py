import pickle
from collections import abc
from pathlib import Path

import httpx

from tricycle.tokeniser import BPETokeniser


class Shakespeare(abc.Sequence):
    url: str = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"  # noqa: E501
    vocab_size: int
    token_path: Path
    raw_data_path: Path
    tokens: list[int]

    def __init__(
        self,
        vocab_size: int,
        token_path: Path | None = None,
        raw_data_path: Path = Path("datasets/shakespeare/raw_data.txt"),
    ):
        if token_path is None:
            token_path = Path(
                f"/home/ben/Documents/Tricycle/datasets/shakespeare/tokens_{vocab_size}.pkl"
            )

        self.vocab_size = vocab_size
        self.raw_data_path = raw_data_path
        self.token_path = token_path

        if not self.token_path.exists():
            self.tokens = self.generate()
            with open(self.token_path, "wb") as f:
                pickle.dump(self.tokens, f)
        else:
            self.tokens = list(self.token_path.read_bytes())

    def download(self):
        """
        Download the shakespeare dataset
        """
        raw_data = httpx.get(self.url).text
        with open(self.raw_data_path, "wb") as f:
            f.write(raw_data.encode("utf-8"))

    def generate(self):
        """
        Download and tokenise the shakespeare dataset
        """
        self.download()
        raw_data = list(self.raw_data_path.read_bytes())
        self.tokeniser = BPETokeniser(self.vocab_size)
        self.tokeniser.train_ints(raw_data)
        return self.tokeniser.tokenise_ints(raw_data)

    def __getitem__(self, idx: int) -> int | list[int]:
        return self.tokens[idx]

    def __len__(self) -> int:
        return len(self.tokens)


if __name__ == "__main__":
    shakespeare = Shakespeare(1024)
