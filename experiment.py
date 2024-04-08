"""
Training script for training a SmolGPT model on the complete
works of shakespeare.

The hyperparams for this model are very much a work in progress
"""

import os
import pickle
from pathlib import Path

import mlflow
import numpy as np
from omegaconf import OmegaConf
from ray import train, tune
from tqdm import tqdm

from inference import get_sample
from tricycle.binary import _shapes_match
from tricycle.dataset import CausalLMDataset
from tricycle.loss import cross_entropy
from tricycle.models import GPT
from tricycle.optimisers import StochasticGradientDescent
from tricycle.scheduler import lr_schedule
from tricycle.tokeniser import BPETokeniser
from tricycle_datasets.shakespeare import Shakespeare

EXPERIMENT_NAME = "SmolGPT:base:find_lr_schedule"

search_space = {
    "model": {
        "embedding_dim": 384,
        "context_window": 256,
        "vocab_size": 1024,
        "n_heads": 6,
        "n_layers": 6,
        "expansion_ratio": 4,
        "activation_fn": "gelu",
        "input_dropout_prob": 0.2,
        "attention_dropout_prob": 0.2,
        "residual_dropout_prob": 0.2,
        "linear_dropout_prob": 0.2,
        "batch_size": 12,
    },
    "train": {
        "max_learning_rate": 1e-4,
        "min_learning_rate": tune.grid_search([1e-4, 1e-5]),
        "warmup_steps": 100,
        "weight_decay": 0,
        "momentum": 0,
        "shuffle": True,
    },
    "mlflow": {
        "tracking_uri": "http://localhost:5000",
        "experiment_name": EXPERIMENT_NAME,
    },
    "experiment": {
        "train_steps": 25_000,
        "valid_steps": 5,
        "valid_every": 25,
        "num_trials": 1,
    },
}


def train_model(config):
    np.random.seed(0)
    config = OmegaConf.create(config)

    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    mlflow.set_experiment(config.mlflow.experiment_name)
    os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"

    model = GPT(config.model)

    current_dir = Path(__file__).parent.absolute()
    raw_data_path = current_dir / "datasets/shakespeare/raw_data.txt"
    tokeniser_path = current_dir / "datasets/shakespeare/tokeniser.pkl"
    token_path = current_dir / "datasets/shakespeare/tokens_1024.pkl"
    shakespeare = Shakespeare(
        vocab_size=config.model.vocab_size,
        raw_data_path=raw_data_path,
        tokeniser_path=tokeniser_path,
        token_path=token_path,
    )

    # train-test split
    n_valid_tokens = (
        config.model.context_window
        + config.experiment.valid_steps * config.model.batch_size
        + 1
    )
    n_train_tokens = (
        config.model.context_window
        + config.experiment.train_steps * config.model.batch_size
        + 1
    )
    assert n_train_tokens + n_valid_tokens < len(
        shakespeare
    ), "Dataset too small"
    train_dataset = (
        CausalLMDataset(
            tokens=shakespeare[:n_train_tokens],
            vocab_size=config.model.vocab_size,
            batch_size=config.model.batch_size,
            context_window=config.model.context_window,
        )
        .batch()
        .to_tensor()
        .to_vector()
    )
    if config.train.shuffle:
        train_dataset = train_dataset.shuffle()
    test_dataset = (
        CausalLMDataset(
            tokens=shakespeare[-n_valid_tokens:],
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
        learning_rate=lr_schedule(
            0,
            max_learning_rate=config.train.max_learning_rate,
            min_learning_rate=config.train.min_learning_rate,
            warmup_steps=config.train.warmup_steps,
            total_steps=config.experiment.train_steps,
        ),
        weight_decay=config.train.weight_decay,
        momentum=config.train.momentum,
    )

    model.to_gpu()

    with mlflow.start_run():
        for key, values in config.items():
            mlflow.log_params({f"{key}/{k}": v for k, v in values.items()})

        for step, (inputs, outputs) in tqdm(
            enumerate(train_dataset), total=config.experiment.train_steps
        ):
            inputs = inputs.to_gpu()
            outputs = outputs.to_gpu()

            logits = model(inputs)
            loss = loss_fn(outputs, logits).from_vector().mean().mean()
            loss.backward()
            model.update(optimiser)

            mlflow.log_metric("train_loss", float(loss.numpy()), step=step)

            # clean up the computational graph
            loss.cleanup()

            # update the lr
            optimiser.learning_rate = lr_schedule(
                step,
                max_learning_rate=config.train.max_learning_rate,
                min_learning_rate=config.train.min_learning_rate,
                warmup_steps=config.train.warmup_steps,
                total_steps=config.experiment.train_steps,
            )
            mlflow.log_metric(
                "learning_rate", optimiser.learning_rate, step=step
            )

            # validation
            if step % config.experiment.valid_every == 0:
                valid_loss = 0
                for inputs, outputs in test_dataset:
                    logits = model(inputs)
                    try:
                        loss = (
                            loss_fn(outputs, logits)
                            .from_vector()
                            .mean()
                            .mean()
                        )
                    except Exception as e:
                        raise Exception(
                            inputs.shape, outputs.shape, logits.shape
                        )

                    valid_loss += float(loss.numpy())
                    loss.cleanup()
                valid_loss /= len(test_dataset)

                sample_text = "HAMLET: To be or not to be"
                assert isinstance(shakespeare.tokeniser, BPETokeniser)
                predicted = get_sample(
                    sample_text, model=model, tokeniser=shakespeare.tokeniser
                )
                model.zero_grad()

                mlflow.log_metric("valid_loss", valid_loss, step=step)
                mlflow.log_text(predicted, f"generated/{step}.txt")
                train.report({"valid_loss": valid_loss})

        # final loss
        valid_loss = 0
        for inputs, outputs in test_dataset:
            logits = model(inputs)
            loss = loss_fn(outputs, logits).from_vector().mean().mean()
            valid_loss += float(loss.numpy())
            loss.cleanup()
        valid_loss /= len(test_dataset)
        mlflow.log_metric("valid_loss", valid_loss, step=len(train_dataset))

        # save the model
        model_dir = Path(
            f"/home/ben/Documents/Tricycle/results/{EXPERIMENT_NAME}/models"
        )
        model_dir.mkdir(parents=True, exist_ok=True)
        with open(
            model_dir / f"lr_{config.train.learning_rate}.pkl", "wb"
        ) as f:
            pickle.dump(model, f)

    return {"valid_loss": valid_loss}


if __name__ == "__main__":
    tuner = tune.Tuner(
        tune.with_resources(
            train_model,
            {"gpu": 1, "cpu": 16},
        ),
        tune_config=tune.TuneConfig(
            metric="valid_loss",
            num_samples=search_space["experiment"]["num_trials"],
        ),
        run_config=train.RunConfig(
            storage_path=Path("results").absolute(),
            name=EXPERIMENT_NAME,
        ),
        param_space=search_space,
    )
    results = tuner.fit()
    results.get_dataframe().to_csv(f"{EXPERIMENT_NAME}_results.csv")
