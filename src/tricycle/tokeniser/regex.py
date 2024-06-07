import json
import os
from multiprocessing import Pool
from pathlib import Path
from warnings import warn

import numpy as np
import regex as re
from numba import njit, prange
from numba.typed import List
from tqdm.auto import tqdm

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
DATA_PATH = Path(__file__).parent / "datasets/shakespeare/raw_data.txt"


@njit()
def count_pairs(data, token_id, counts):
    for i in range(len(data) - 1):
        left, right = data[i], data[i + 1]
        counts[left * (token_id + 1) + right] += 1
    return counts


@njit
def replace_pair(
    data: np.ndarray, pair: tuple[int, int], token_id: int
) -> np.ndarray:
    """
    Replace every occurrence of `pair` with `token_id` for a single array
    """
    if len(data) == 1:
        return data
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


@njit(parallel=False)
def replace_all_pairs(data: np.ndarray, pair: tuple[int, int], token_id: int):
    for i in prange(len(data)):
        data[i] = replace_pair(data[i], pair, token_id)
    return data


@njit(parallel=False)
def count_all_pairs(data, token_id):
    counts = np.zeros((token_id + 1) ** 2, dtype=np.int32)
    for i in range(len(data)):
        chunk = data[i]
        counts = count_pairs(chunk, token_id, counts)
    return counts


@njit
def flatten(int_array, size=1024, scaling_factor=2):
    flat = np.zeros(size, dtype=np.int32)
    idx = 0

    for chunk in int_array:
        for token in chunk:
            flat[idx] = token
            idx += 1
            if idx == size:
                size = int(size * scaling_factor)
                empty = np.zeros(size, dtype=np.int32)
                empty[: size // 2] = flat
                flat = empty
    return flat[:idx]


class RegexTokeniser:
    vocab_size: int
    merges: dict[tuple[int, int | None], int]
    pairs: list[tuple[int, int | None]]

    # we cant have less than the number of possible single bytes
    MIN_TOKENS = 256
    PATTERN = re.compile(GPT4_SPLIT_PATTERN)

    def __init__(self, vocab_size: int):
        assert (
            vocab_size >= self.MIN_TOKENS
        ), f"vocab_size must be >= {self.MIN_TOKENS}"
        self.vocab_size = vocab_size

        # initialise our pairs and merges with single byte tokens
        self.pairs = [(idx, None) for idx in range(self.MIN_TOKENS)]
        self.merges = {(idx, None): idx for idx in range(self.MIN_TOKENS)}
        self.vocab = [idx.to_bytes(1, "big") for idx in range(self.MIN_TOKENS)]

    def save(self, path):
        data = {
            "vocab_size": self.vocab_size,
            "pairs": self.pairs,
            "merges": self.merges,
            "vocab": self.vocab,
        }
        with open(path, "wb") as f:
            json.dump(data, f)

    def chunk_single_file(self, text: str) -> list[np.ndarray]:
        chunks = self.PATTERN.findall(text)
        return [
            np.array(list(chunk.encode("utf-8")), dtype=np.int32)
            for chunk in chunks
        ]

    def chunk(self, documents: list[str], chunk_size=16):
        with Pool(os.cpu_count()) as pool:
            result = list(
                tqdm(
                    pool.imap_unordered(
                        self.chunk_single_file, documents, chunk_size
                    ),
                    total=len(documents),
                    desc="Chunking",
                )
            )
        return List(tqdm(result, desc="converting to numba list"))

    def train_chunks(
        self, chunks: list[np.ndarray], loading_bar: bool = False
    ):
        token_ids = range(self.MIN_TOKENS, self.vocab_size)
        if loading_bar:
            token_ids = tqdm(token_ids, desc="Training tokeniser")

        for token_id in token_ids:
            # find the most common pair of tokens
            most_common_pair = self.most_common_pair(
                count_all_pairs(chunks, token_id), token_id
            )
            if most_common_pair is None:
                break

            # replace every occurrence of the pair with the new token
            int_array = replace_all_pairs(chunks, most_common_pair, token_id)

            # store the new pair and token
            self.merges[most_common_pair] = token_id
            self.pairs.append(most_common_pair)
            left, right = most_common_pair
            self.vocab.append(self.vocab[left] + self.vocab[right])

        if len(self.pairs) != self.vocab_size:
            warn(f"Expected {self.vocab_size} pairs, got {len(self.pairs)}")
        return self

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


if __name__ == "__main__":
    sample_text = DATA_PATH.read_text()
    tokeniser = Tokeniser(1000)
    tokens = tokeniser.encode(sample_text, loading_bar=True)
