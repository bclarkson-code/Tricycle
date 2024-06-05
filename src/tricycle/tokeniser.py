import pickle
from pathlib import Path
from typing import List, Tuple
from warnings import warn

import numpy as np
from numba import jit, prange
from tqdm.auto import tqdm


class BPETokeniser:
    """
    A simple byte pair encoding tokeniser
    """

    # we cant have less than the number of possible single bytes
    MIN_TOKENS = 256

    vocab_size: int
    merges: dict[tuple[int, int | None], int]
    pairs: list[tuple[int, int | None]]

    def __init__(self, vocab_size: int):
        assert (
            vocab_size >= self.MIN_TOKENS
        ), f"vocab_size must be >= {self.MIN_TOKENS}"
        self.vocab_size = vocab_size

        # initialise our pairs and merges with single byte tokens
        self.pairs = [(idx, None) for idx in range(self.MIN_TOKENS)]
        self.merges = {(idx, None): idx for idx in range(self.MIN_TOKENS)}
        self.vocab = [idx.to_bytes(1, "big") for idx in range(self.MIN_TOKENS)]

    def count_pairs(self, data: list[int]):
        counts = {}
        for left, right in zip(data[:-1], data[1:]):
            counts[(left, right)] = counts.get((left, right), 0) + 1
        return counts

    def most_common_pair(
        self, counts: dict[tuple[int, int], int]
    ) -> tuple[int, int] | None:
        """
        Return the most common pair
        """
        if not counts:
            return None
        most_common = max(counts, key=counts.get)
        return None if counts.get(most_common) == 1 else most_common

    def replace_pair(
        self, data: list[int], pair: tuple[int, int], token_id: int
    ) -> list[int]:
        """
        Replace every occurrence of `pair` with `token_id`
        """
        out = []
        skip_next = False
        for left, right in zip(data[:-1], data[1:]):
            if skip_next:
                skip_next = False
                continue

            if (left, right) == pair:
                skip_next = True
                out.append(token_id)
            else:
                out.append(left)
        if not skip_next and data[1:]:
            out.append(right)

        return out

    def train_ints(self, int_array: list[int], loading_bar=False):
        """
        Train the tokeniser on an array of ints
        """
        token_ids = range(self.MIN_TOKENS, self.vocab_size)
        if loading_bar:
            token_ids = tqdm(token_ids, desc="Training tokeniser")
        for token_id in token_ids:
            most_common_pair = self.most_common_pair(
                self.count_pairs(int_array)
            )
            if most_common_pair is None:
                break

            int_array = self.replace_pair(
                int_array, most_common_pair, token_id
            )
            self.merges[most_common_pair] = token_id
            self.pairs.append(most_common_pair)
            left, right = most_common_pair
            self.vocab.append(self.vocab[left] + self.vocab[right])

        self.tokens = int_array

        if len(self.pairs) != self.vocab_size:
            warn(f"Expected {self.vocab_size} pairs, got {len(self.pairs)}")
        return self

    def train(self, text: str):
        """
        Train the tokeniser on a string
        """
        as_bytes = text.encode("utf-8")
        as_ints = list(as_bytes)
        return self.train_ints(as_ints)

    def tokenise_ints(
        self, int_array: list[int], loading_bar=False
    ) -> list[int]:
        """
        Tokenise an array of ints
        """
        ints = self.merges.items()
        if loading_bar:
            ints = tqdm(ints, desc="tokenising", total=len(ints))
        for pair, token_id in ints:
            if pair[1] is None:
                continue
            int_array = self.replace_pair(int_array, pair, token_id)
        return int_array

    def encode(self, text: str) -> list[int]:
        """
        Tokenise a string
        """
        as_bytes = text.encode("utf-8")
        as_ints = list(as_bytes)

        return self.tokenise_ints(as_ints)

    def decode(self, tokens: list[int] | int) -> str:
        """
        Convert tokens into a string
        """
        if not isinstance(tokens, list):
            tokens = [tokens]

        decoded = b""
        for token in tokens:
            decoded += self.vocab[token]
        return decoded.decode("utf-8", errors="replace")

    def save(self, path: str | Path):
        with open(path, "wb") as f:
            state = {
                "vocab_size": self.vocab_size,
                "merges": self.merges,
                "pairs": self.pairs,
            }
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: str | Path):
        with open(path, "rb") as f:
            state = pickle.load(f)
            result = cls(
                state["vocab_size"],
            )
            result.merges = state["merges"]
            result.pairs = state["pairs"]
            return result


@jit(nopython=True, cache=True)
def replace_pair(
    data: np.ndarray, pair: Tuple[int, int], token_id: int
) -> np.ndarray:
    """
    Replace every occurrence of `pair` with `token_id`
    """
    new = 0
    old = 0

    while old < len(data) - 1:
        left = data[old]
        right = data[old + 1]

        if (left, right) == pair:
            data[new] = token_id
            old += 1
        else:
            data[new] = left
        new += 1
        old += 1

    # handle final id not being a match
    if old == len(data) - 1 and old != 0:
        data[new] = data[old]
        new += 1

    return data[:new]


@jit("int32[:](int32[:], int32)", nopython=True, parallel=False)
def count_pairs(data: np.ndarray, token_id: int) -> np.ndarray:
    counts = np.zeros((token_id + 1) ** 2, dtype=np.int32)
    for i in range(len(data) - 1):
        left, right = data[i], data[i + 1]
        counts[(left * (token_id + 1)) + right] += 1

    return counts


class BPETokeniserNumba:
    """
    A simple byte pair encoding tokeniser
    """

    # we cant have less than the number of possible single bytes
    MIN_TOKENS = 256

    vocab_size: int
    merges: dict[tuple[int, int | None], int]
    pairs: list[tuple[int, int | None]]

    def __init__(self, vocab_size: int):
        assert (
            vocab_size >= self.MIN_TOKENS
        ), f"vocab_size must be >= {self.MIN_TOKENS}"
        self.vocab_size = vocab_size

        # initialise our pairs and merges with single byte tokens
        self.pairs = [(idx, None) for idx in range(self.MIN_TOKENS)]
        self.merges = {(idx, None): idx for idx in range(self.MIN_TOKENS)}
        self.vocab = [idx.to_bytes(1, "big") for idx in range(self.MIN_TOKENS)]
        self.type_ = "numba"

    def count_pairs(self, data: np.ndarray, token_id: int) -> np.ndarray:
        return count_pairs(data, token_id)

    def replace_pair(
        self, data: np.ndarray, pair: tuple[int, int], token_id: int
    ) -> np.ndarray:
        return replace_pair(data=data, pair=pair, token_id=token_id)

    def most_common_pair(
        self, counts: np.ndarray, token_id: int
    ) -> tuple[int, int] | None:
        """
        Return the most common pair
        """
        most_common_idx = np.argmax(counts)

        # check if there are no more repeated pairs
        if counts[most_common_idx] in {0, 1}:
            return None

        left = most_common_idx // (token_id + 1)
        right = most_common_idx % (token_id + 1)

        return left, right

    def train_ints(self, int_array: np.ndarray, loading_bar=False):
        """
        Train the tokeniser on an array of ints
        """
        token_ids = range(self.MIN_TOKENS, self.vocab_size)
        if loading_bar:
            token_ids = tqdm(token_ids, desc="Training tokeniser")
        for token_id in token_ids:
            # find the most common pair of tokens
            most_common_pair = self.most_common_pair(
                self.count_pairs(int_array, token_id), token_id
            )
            if most_common_pair is None:
                break

            # replace every occurrence of the pair with the new token
            int_array = self.replace_pair(
                int_array, most_common_pair, token_id
            )

            # store the new pair and token
            self.merges[most_common_pair] = token_id
            self.pairs.append(most_common_pair)
            left, right = most_common_pair
            self.vocab.append(self.vocab[left] + self.vocab[right])

        self.tokens = int_array

        if len(self.pairs) != self.vocab_size:
            warn(f"Expected {self.vocab_size} pairs, got {len(self.pairs)}")
        return self

    def train(self, text: str):
        """
        Train the tokeniser on a string
        """
        as_bytes = text.encode("utf-8")
        as_ints = np.array(list(as_bytes), dtype=np.int32)
        return self.train_ints(as_ints)

    def tokenise_ints(
        self, int_array: np.ndarray, loading_bar=False
    ) -> np.ndarray:
        """
        Tokenise an array of ints
        """
        if not isinstance(int_array, np.ndarray):
            int_array = np.array(int_array, dtype=np.int32)

        ints = self.merges.items()
        if loading_bar:
            ints = tqdm(ints, desc="tokenising", total=len(ints))
        for pair, token_id in ints:
            if pair[1] is None:
                continue
            int_array = self.replace_pair(int_array, pair, token_id)
        return int_array

    def encode(self, text: str) -> np.ndarray:
        """
        Tokenise a string
        """
        as_bytes = text.encode("utf-8")
        as_ints = np.array(list(as_bytes))

        return self.tokenise_ints(as_ints)

    def decode(self, tokens: np.ndarray | int) -> str:
        """
        Convert tokens into a string
        """
        if not isinstance(tokens, np.ndarray):
            tokens = np.array([tokens])

        decoded = b""
        for token in tokens:
            decoded += self.vocab[token]
        return decoded.decode("utf-8", errors="replace")

    def save(self, path: str | Path):
        with open(path, "wb") as f:
            state = {
                "vocab_size": self.vocab_size,
                "merges": self.merges,
                "pairs": self.pairs,
            }
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: str | Path):
        with open(path, "rb") as f:
            state = pickle.load(f)
            result = cls(
                state["vocab_size"],
            )
            result.merges = state["merges"]
            result.pairs = state["pairs"]
            return result


class BPETokeniserNumbaClass:
    """
    A simple byte pair encoding tokeniser
    """

    # we cant have less than the number of possible single bytes
    MIN_TOKENS = 256

    vocab_size: int
    merges: dict[tuple[int, int | None], int]
    pairs: list[tuple[int, int | None]]

    def __init__(self, vocab_size: int):
        assert (
            vocab_size >= self.MIN_TOKENS
        ), f"vocab_size must be >= {self.MIN_TOKENS}"
        self.vocab_size = vocab_size

        # initialise our pairs and merges with single byte tokens
        self.pairs = [(idx, None) for idx in range(self.MIN_TOKENS)]
        self.merges = {(idx, None): idx for idx in range(self.MIN_TOKENS)}
        self.vocab = [idx.to_bytes(1, "big") for idx in range(self.MIN_TOKENS)]
        self.type_ = "numba"

    def count_pairs(self, data: np.ndarray, token_id: int) -> np.ndarray:

        match self.type_:
            case "2d":
                return count_pairs_2d(data, token_id)
            case _:
                return count_pairs(data, token_id)

    def replace_pair(
        self, data: np.ndarray, pair: tuple[int, int], token_id: int
    ) -> np.ndarray:
        return replace_pair(data=data, pair=pair, token_id=token_id)

    def most_common_pair(
        self, counts: np.ndarray, token_id: int
    ) -> tuple[int, int] | None:
        """
        Return the most common pair
        """
        most_common_idx = np.argmax(counts)

        # check if there are no more repeated pairs
        if counts[most_common_idx] in {0, 1}:
            return None

        left = most_common_idx // (token_id + 1)
        right = most_common_idx % (token_id + 1)

        return left, right

    def train_ints(self, int_array: np.ndarray, loading_bar=False):
        """
        Train the tokeniser on an array of ints
        """
        token_ids = range(self.MIN_TOKENS, self.vocab_size)
        if loading_bar:
            token_ids = tqdm(token_ids, desc="Training tokeniser")
        for token_id in token_ids:
            most_common_pair = self.most_common_pair(
                self.count_pairs(int_array, token_id), token_id
            )
            if most_common_pair is None:
                break

            int_array = replace_pair(int_array, most_common_pair, token_id)
            self.merges[most_common_pair] = token_id
            self.pairs.append(most_common_pair)
            left, right = most_common_pair
            self.vocab.append(self.vocab[left] + self.vocab[right])

        self.tokens = int_array

        if len(self.pairs) != self.vocab_size:
            warn(f"Expected {self.vocab_size} pairs, got {len(self.pairs)}")
        return self

    def train(self, text: str):
        """
        Train the tokeniser on a string
        """
        as_bytes = text.encode("utf-8")
        as_ints = np.array(list(as_bytes), dtype=np.int32)
        return self.train_ints(as_ints)

    def tokenise_ints(
        self, int_array: list[int], loading_bar=False
    ) -> list[int]:
        """
        Tokenise an array of ints
        """
        ints = self.merges.items()
        if loading_bar:
            ints = tqdm(ints, desc="tokenising", total=len(ints))
        for pair, token_id in ints:
            if pair[1] is None:
                continue
            int_array = self.replace_pair(int_array, pair, token_id)
        return int_array

    def encode(self, text: str) -> list[int]:
        """
        Tokenise a string
        """
        as_bytes = text.encode("utf-8")
        as_ints = list(as_bytes)

        return self.tokenise_ints(as_ints)

    def decode(self, tokens: list[int] | int) -> str:
        """
        Convert tokens into a string
        """
        if not isinstance(tokens, list):
            tokens = [tokens]

        decoded = b""
        for token in tokens:
            decoded += self.vocab[token]
        return decoded.decode("utf-8", errors="replace")

    def save(self, path: str | Path):
        with open(path, "wb") as f:
            state = {
                "vocab_size": self.vocab_size,
                "merges": self.merges,
                "pairs": self.pairs,
            }
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: str | Path):
        with open(path, "rb") as f:
            state = pickle.load(f)
            result = cls(
                state["vocab_size"],
            )
            result.merges = state["merges"]
            result.pairs = state["pairs"]
            return result
