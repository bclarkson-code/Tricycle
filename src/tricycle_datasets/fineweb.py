import os
from collections import abc
from pathlib import Path
from typing import Literal

import numpy as np
import tiktoken
from tqdm.auto import tqdm

# import datasets
# from datasets import load_dataset

N_CORES = os.cpu_count()
SAVE_DIR = Path("datasets/fineweb")
SAVE_DIR.mkdir(exist_ok=True, parents=True)
DTYPE = np.uint16


tokeniser = tiktoken.get_encoding("gpt2")


def tokenise_document(example):
    """
    Tokenise a single document from the dataset
    """
    ids = tokeniser.encode_ordinary(
        example["text"]
    )  # encode_ordinary ignores any special tokens
    ids.append(tokeniser.eot_token)  # add the end of text token
    out = {"ids": ids, "len": len(ids)}
    return out


def prepare_data():
    """
    Download and tokenise the coreparrot dataset. Note, this script is adapted
    from Andjrey Karpathy's NanoGPT:
    https://github.com/karpathy/nanoGPT/blob/master/data/openwebtext/prepare.py

    For now, my tokeniser is too slow for large datasets so i'm using openai's
    tiktokeniser
    """
    datasets.disable_caching()
    dataset = load_dataset(
        "HuggingFaceFW/fineweb", name="sample-10BT", split="train"
    )
    split_dataset = dataset.train_test_split(
        test_size=0.0005, seed=2357, shuffle=True
    )
    split_dataset["valid"] = split_dataset.pop(
        "test"
    )  # rename the test split to val

    # tokenise the dataset
    tokenised = split_dataset.map(
        tokenise_document,
        remove_columns=["text"],
        desc="Tokenising",
        num_proc=N_CORES,
    )

    # concatenate all the ids in each dataset into one large file we can use
    # for training
    for split, dset in tokenised.items():
        filename = SAVE_DIR / f"{split}.bin"

        n_tokens = np.sum(dset["len"])
        print(f"Found: {n_tokens} {split} tokens")

        arr = np.memmap(filename, dtype=DTYPE, mode="w+", shape=(n_tokens,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(
            range(total_batches), desc=f"writing {filename}"
        ):
            # Batch together samples for faster write
            batch = dset.shard(
                num_shards=total_batches, index=batch_idx, contiguous=True
            ).with_format("numpy")
            arr_batch = np.concatenate(batch["ids"])
            # Write into mmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()


class FineWeb(abc.Sequence):
    vocab_size: int
    token_path: Path
    tokeniser_string: str = "gpt2"
    tokens: np.ndarray

    def __init__(
        self,
        vocab_size: int,
        split: Literal["train"] | Literal["valid"],
        token_path: Path | None = None,
    ):
        self.vocab_size = vocab_size

        self.tokeniser = tiktoken.get_encoding(self.tokeniser_string)
        if self.tokeniser.max_token_value != vocab_size:
            raise ValueError(
                "Expected tokeniser.max_token_value == vocab_size. Found "
                f"{self.tokeniser.max_token_value=}, {vocab_size=}"
            )

        if token_path is None:
            self.token_path = SAVE_DIR / f"{split}.bin"
        else:
            self.token_path = token_path

        if not self.token_path.exists():
            prepare_data()

        assert self.token_path.exists()
        self.tokens = np.memmap(self.token_path, dtype=DTYPE, mode="r")

    def __getitem__(self, key):
        return self.tokens[key]

    def __len__(self):
        return len(self.tokens)

    def encode(self, *args):
        return self.tokeniser.encode_ordinary(*args)

    def decode(self, *args):
        return self.tokeniser.decode(*args)


if __name__ == "__main__":
    prepare_data()
