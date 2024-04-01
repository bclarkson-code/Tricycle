"""
Training script for training a SmolGPT model on the complete
works of shakespeare.

The hyperparams for this model are very much a work in progress
"""

import os

import mlflow
from omegaconf import OmegaConf
from ray import tune
from tqdm import tqdm

from tricycle.activation import GeLU
from tricycle.dataset import CausalLMDataset
from tricycle.loss import cross_entropy
from tricycle.models import GPT
from tricycle.optimisers import StochasticGradientDescent
from tricycle_datasets.shakespeare import Shakespeare

search_space = {
    "model": {
        "embedding_dim": 128,
        "context_window": 64,
        "vocab_size": 1024,
        "n_heads": 2,
        "n_layers": 1,
        "expansion_ratio": 4,
        "activation_fn": "gelu",
        "input_dropout_prob": 0,
        "attention_dropout_prob": 0,
        "residual_dropout_prob": 0,
        "linear_dropout_prob": 0,
    },
    "train": {
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "weight_decay": 0,
        "momentum": 0,
        "batch_size": 32,
        "n_steps": 10_000,
    },
    "mlflow": {
        "mlflow_enabled": True,
        "mlflow_tracking_uri": "http://localhost:5000",
        "mlflow_experiment_name": "SmolGPT:base:find_learning_rate",
    },
    "experiment": {
        "train_steps": 10_000,
        "valid_steps": 10,
        "valid_every": 100,
    },
}


def train(config):
    config = OmegaConf.create(config)

    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    mlflow.set_experiment(config.mlflow.experiment_name)
    os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"

    match config.model.activation_fn:
        case "gelu":
            config.model.activation_fn = GeLU()
        case _:
            raise ValueError(
                f"Unknown activation function: {config.model.activation_fn}"
            )

    model = GPT(config.model)

    tokens = Shakespeare(vocab_size=config.model.vocab_size)

    # train-test split
    n_valid_tokens = config.experiment.valid_steps * config.model.batch_size
    n_train_tokens = config.experiment.train_steps * config.model.batch_size
    assert n_train_tokens + n_valid_tokens < len(tokens), "Dataset too small"
    train_dataset = (
        CausalLMDataset(
            tokens=tokens[:n_train_tokens],
            vocab_size=config.model.vocab_size,
            batch_size=config.model.batch_size,
            context_window=config.model.context_window,
        )
        .batch()
        .to_tensor()
        .to_vector()
    )
    test_dataset = (
        CausalLMDataset(
            tokens=tokens[-n_valid_tokens:],
            vocab_size=config.model.vocab_size,
            batch_size=config.model.batch_size,
            context_window=config.model.context_window,
        )
        .batch()
        .to_tensor()
        .to_vector()
    )
    loss_fn = cross_entropy
    optimiser = StochasticGradientDescent(
        learning_rate=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
        momentum=config.train.momentum,
    )

    with mlflow.start_run():
        mlflow.log_params(config)

        for step, (inputs, outputs) in tqdm(
            enumerate(train_dataset), total=config.train.n_steps
        ):
            logits = model(inputs)
            loss = loss_fn(outputs, logits).from_vector().mean().mean()
            loss.backward()
            model.update(optimiser)

            mlflow.log_metric("loss", loss.item())

            # clean up the computational graph
            loss.cleanup()

            # validation
            if step % config.experiment.valid_every == 0:
                valid_loss = 0
                for inputs, outputs in test_dataset:
                    logits = model(inputs)
                    loss = loss_fn(outputs, logits).from_vector().mean().mean()
                    valid_loss += float(loss)
                    loss.cleanup()
                valid_loss /= len(test_dataset)

                mlflow.log_metric("valid_loss", valid_loss)
    return
