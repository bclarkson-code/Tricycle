import pickle
from collections import abc
from pathlib import Path

import httpx

from tricycle.tokeniser import BPETokeniser


class Shakespeare(abc.Sequence):
    url: str = (
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"  # noqa: E501
    )
    vocab_size: int
    token_path: Path
    raw_data_path: Path
    tokens: list[int]

    def __init__(
        self,
        vocab_size: int,
        token_path: Path | None = None,
        raw_data_path: Path = Path("datasets/shakespeare/raw_data.txt"),
        tokeniser_path: Path = Path("datasets/shakespeare/tokeniser.pkl"),
    ):
        if token_path is None:
            token_path = Path(f"datasets/shakespeare/tokens_{vocab_size}.pkl")

        self.vocab_size = vocab_size
        self.raw_data_path = raw_data_path
        self.token_path = token_path
        self.tokeniser_path = tokeniser_path

        if self.tokeniser_path.exists():
            with open(self.tokeniser_path, "rb") as f:
                self.tokeniser = pickle.load(f)
        else:
            self.tokeniser = None

        if not self.token_path.exists():
            self.tokeniser = self.generate()
            self.tokeniser_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.tokeniser_path, "wb") as f:
                pickle.dump(self.tokeniser, f)

            self.tokens = self.tokeniser.tokens
            self.token_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.token_path, "wb") as f:
                pickle.dump(self.tokens, f)
        else:
            with open(self.token_path, "rb") as f:
                self.tokens = pickle.load(f)

    def download(self):
        """
        Download the shakespeare dataset
        """
        raw_data = httpx.get(self.url).text
        self.raw_data_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.raw_data_path, "w") as f:
            f.write(raw_data)

    def generate(self) -> BPETokeniser:
        """
        Download and tokenise the shakespeare dataset
        """
        self.download()
        raw_data = list(self.raw_data_path.read_bytes())
        if self.tokeniser is None:
            self.tokeniser = BPETokeniser(self.vocab_size)
        return self.tokeniser.train_ints(raw_data, loading_bar=True)

    def __getitem__(self, idx: int) -> int | list[int]:
        return self.tokens[idx]

    def __len__(self) -> int:
        return len(self.tokens)


class ShakespeareChar(abc.Sequence):
    url: str = (
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"  # noqa: E501
    )
    vocab_size: int
    raw_data_path: Path
    chars: list[int]

    def __init__(
        self,
        raw_data_path: Path = Path("datasets/shakespeare/raw_data.txt"),
    ):
        self.raw_data_path = raw_data_path
        self.chars = self.generate()
        self.vocab_size = len(set(self.chars))

    def encode(self, chars: list[int] | str):
        if isinstance(chars, str):
            chars = [ord(i) for i in chars]
        return [self.char_ids[c] for c in chars]

    def decode(self, char_ids: list[int]):
        inv_char_ids = {i: c for c, i in self.char_ids.items()}
        return [inv_char_ids[i] for i in char_ids]

    def download(self):
        """
        Download the shakespeare dataset
        """
        raw_data = httpx.get(self.url).text
        self.raw_data_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.raw_data_path, "w") as f:
            f.write(raw_data)

    def generate(self) -> list[int]:
        """
        Download and tokenise the shakespeare dataset
        """
        if not self.raw_data_path.exists():
            self.download()

        raw_data = list(self.raw_data_path.read_bytes())
        self.char_ids = {c: i for i, c in enumerate(set(raw_data))}
        return self.encode(raw_data)

    def __getitem__(self, idx: int) -> int | list[int]:
        return self.chars[idx]

    def __len__(self) -> int:
        return len(self.chars)


if __name__ == "__main__":
    shakespeare = Shakespeare(1024)
