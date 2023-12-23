import logging
import math
import os
import warnings
from argparse import ArgumentParser, Namespace
from pathlib import Path

import datasets
import humanize
import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from lightning.pytorch.loggers import MLFlowLogger
from tokenizers import Tokenizer, trainers
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast

L.seed_everything(42)
torch.set_float32_matmul_precision("high")


def cosine_decay(it, warmup_iters=10_000, lr_decay_iters=100_000, min_frac=0.1):
    # warmup
    if it < warmup_iters:
        return it / warmup_iters
    # finished decay
    if it > lr_decay_iters:
        return min_frac
    # decay
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_frac + coeff * (1 - min_frac)


class Word2Vec(L.LightningModule):
    def __init__(
        self,
        vocab_size: int,
        embedding_size: int = 64,
        hidden_size: int = 256,
        loss_fn: str = "cross_entropy",
        lr: float = 1e-3,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

        self.encoder = nn.Sequential(
            nn.EmbeddingBag(self.vocab_size, self.hidden_size, mode="mean"),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.embedding_size),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.embedding_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.vocab_size),
        )

        self.loss_fn = self._load_loss_fn(loss_fn)

    def _load_loss_fn(self, loss_fn_string):
        match loss_fn_string:
            case "cross_entropy":
                return nn.NLLLoss()
            case _:
                raise NotImplementedError(f"Unknown loss {loss_fn_string}")

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def training_step(self, batch, _):
        return self._calculate_loss(batch, "train_loss")

    def validation_step(self, batch, _):
        self._calculate_loss(batch, "val_loss")

    def test_step(self, batch, _):
        self._calculate_loss(batch, "test_loss")

    def _calculate_loss(self, batch, arg1):
        inputs, targets = batch
        outputs = self(inputs)
        outputs = nn.LogSoftmax(dim=1)(outputs)
        loss = self.loss_fn(outputs, targets)
        self.log(arg1, loss, sync_dist=True)
        return loss

    def predict_step(self, batch, _):
        inputs, _ = batch
        return self(inputs)

    def configure_optimizers(self):
        optimiser = torch.optim.SGD(
            self.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        return {
            "optimizer": optimiser,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(
                    optimiser, lr_lambda=cosine_decay
                ),
                "interval": "step",
                "frequency": 1,
            },
        }


class WikipediaDataset(Dataset):
    def __init__(
        self,
        tokeniser_path="wikipedia_tokeniser.json",
        save_path="wikipedia.bin",
        vocab_size=50_000,
        window_size=5,
        pad_token_id=50_000,
        split="all",
        valid_size=1_000_000,
        test_size=1_000_000,
    ):
        print(f"creating new dataset: { split }")
        self.vocab_size = vocab_size
        self.window_size = window_size
        self.tokeniser_path = Path(tokeniser_path)
        self.save_path = Path(save_path)

        ds = self.load_dataset(self.save_path)
        match split:
            case "all":
                self.start, self.end = 0, len(ds)
            case "train":
                self.start, self.end = 0, len(ds) - (valid_size + test_size)
            case "test":
                self.start, self.end = len(ds) - test_size, len(ds)
            case "valid":
                self.start, self.end = (
                    len(ds) - (valid_size + test_size),
                    len(ds) - test_size,
                )
            case _:
                warnings.warn(
                    f"Could not parse split: {split}. feaulting to entire dataset"
                )
                self.start, self.end = 0, len(ds)

    def preprocess(self, ds):
        self.tokeniser = self.load_tokeniser(self.tokeniser_path)
        ds = ds.map(self.tokenise, num_proc=os.cpu_count(), desc="Tokenising")
        ds = ds.map(self.squeeze, num_proc=os.cpu_count(), desc="Squeezing")
        ds = ds.map(self.get_length, num_proc=os.cpu_count(), desc="Finding Lengths")
        self.combine_docs()

    def get_length(self, doc):
        return {"len": len(doc["input_ids"])}

    def squeeze(self, doc):
        return {"input_ids": doc["input_ids"][0]}

    def tokenise(self, batch):
        return self.tokeniser(batch["text"], return_tensors="pt")

    def load_tokeniser(self, tokeniser_path):
        if tokeniser_path.exists():
            tokeniser = Tokenizer.from_file(tokeniser_path.absolute().as_posix())
        else:
            ds = datasets.load_dataset(
                "wikipedia",
                "20220301.en",
                split="train",  # train is the only split available
            )
            tokeniser = self.train_tokeniser(ds)
        return PreTrainedTokenizerFast(
            tokenizer_object=tokeniser,
            pad_token="<|pad|>",
            bos_token="<|endoftext|>",
            eos_token="<|endoftext|>",
        )

    def train_tokeniser(self, ds):
        tokeniser = Tokenizer(BPE(unk_token="[UNK]"))
        trainer = trainers.BpeTrainer(
            vocab_size=50_000, special_tokens=["<|endoftext|>"]
        )
        tokeniser.pre_tokenizer = ByteLevel(add_prefix_space=False)
        tokeniser.train_from_iterator(ds["text"], trainer)
        tokeniser.save(self.tokeniser_path.absolute().as_posix())
        return tokeniser

    def load_dataset(self, save_path):
        if not save_path.exists():
            ds = datasets.load_dataset(
                "wikipedia",
                "20220301.en",
                split="train",  # train is the only split available
            ).shuffle(seed=42)
            self.preprocess(ds)
        ds = np.memmap(save_path, dtype=np.uint16, mode="r")
        return ds

    def __getitem__(self, idx: int):
        """
        Get a bag of tokens and the label for the bag

        Ignore the first and last few tokens so that
        our windows never overlap with the ends of the array
        (we have 4.5 Billion tokens, losing 10 is ok
        """
        ds = self.load_dataset(self.save_path)
        idx += self.start + self.window_size

        all_indices = range(idx - self.window_size, idx + self.window_size + 1)
        indices = [i for i in all_indices if i != idx]

        bag_of_words = torch.from_numpy(ds[indices].astype(np.int64))
        value = torch.tensor(ds[idx].astype(np.int64))
        return bag_of_words, value

    def __len__(self):
        return self.end - self.start

    def combine_docs(self, ds):
        total_tokens = sum(tqdm(ds["len"], desc="Counting Tokens"))
        human_n_tokens = humanize.intword(total_tokens)
        print(f"Found {human_n_tokens} tokens")

        combined = np.memmap(
            self.save_path, dtype=np.uint16, mode="w+", shape=(total_tokens,)
        )
        n_batches = 1024
        arr_idx = 0
        for batch_idx in tqdm(range(n_batches), desc="Concatenating"):
            batch = ds.shard(num_shards=n_batches, index=batch_idx)
            batch = np.concatenate(batch["input_ids"])
            combined[arr_idx : arr_idx + len(batch)] = batch
            arr_idx += len(batch)
        combined.flush()


def main(args: Namespace):
    train_ds = WikipediaDataset(
        split="train", test_size=args.test_size, valid_size=args.validation_size
    )
    valid_ds = WikipediaDataset(
        split="valid", test_size=args.test_size, valid_size=args.validation_size
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        num_workers=args.num_train_workers,
        pin_memory=True,
    )
    valid_dl = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        num_workers=args.num_valid_workers,
    )
    model = Word2Vec(train_ds.vocab_size)

    root_dir = Path("word2vec")
    os.makedirs(root_dir, exist_ok=True)
    mlflow_logger = MLFlowLogger(
        experiment_name="word2vec_wikipedia",
        tracking_uri="http://0.0.0.0:5000",
    )

    trainer = L.Trainer(
        default_root_dir=root_dir,
        callbacks=[
            ModelCheckpoint(save_weights_only=True, mode="min", monitor="val_loss"),
            EarlyStopping(monitor="val_loss", mode="min", patience=3),
            LearningRateMonitor(logging_interval="step"),
        ],
        logger=mlflow_logger,
        accelerator=args.accelerator,
        devices=args.devices,
        val_check_interval=1000,
        max_epochs=1,
        gradient_clip_val=5,
        # strategy="ddp_notebook",
        precision="16-mixed",
    )

    trainer.fit(model, train_dl, valid_dl)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--accelerator", default="gpu")
    parser.add_argument("--devices", default=1)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--num_train_workers", default=os.cpu_count())
    parser.add_argument("--num_valid_workers", default=os.cpu_count())
    parser.add_argument("--test_size", default=100_000)
    parser.add_argument("--validation_size", default=100_000)
    parser.add_argument("--vocab_size", default=50_000)
    args = parser.parse_args()

    main(args)
