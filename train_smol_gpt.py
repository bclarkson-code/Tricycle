"""
Training script for training a SmolGPT model on the complete
works of shakespeare.

The hyperparams for this model are very much a work in progress
"""

import datetime
import gc
import os
import pickle

import mlflow
from tqdm import tqdm

from tricycle.configs import SmolGPTConfig
from tricycle.dataset import CausalLMDataset
from tricycle.loss import cross_entropy
from tricycle.models import GPT
from tricycle.optimisers import StochasticGradientDescent
from tricycle_datasets.shakespeare import Shakespeare

config = SmolGPTConfig()
model = GPT(config)

if config.mlflow_enabled:
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment(config.mlflow_experiment_name)
    os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"


tokens = Shakespeare(vocab_size=config.vocab_size)
dataset = (
    CausalLMDataset(
        tokens=tokens,
        vocab_size=config.vocab_size,
        batch_size=config.batch_size,
        context_window=config.context_window,
    )
    .batch()
    .to_tensor()
    .to_vector()
)
loss_fn = cross_entropy
optimiser = StochasticGradientDescent(
    learning_rate=config.learning_rate,
    weight_decay=config.weight_decay,
    momentum=config.momentum,
)

if config.mlflow_enabled:
    with mlflow.start_run():
        mlflow.log_params(config.__dict__)

        try:
            for step, (inputs, outputs) in tqdm(
                enumerate(dataset), total=len(dataset)
            ):
                logits = model(inputs)
                loss = loss_fn(outputs, logits).from_vector().mean().mean()
                loss.backward()
                model.update(optimiser)
                mlflow.log_metric("loss", loss, step=step)
        # save before crashing
        except Exception as e:
            with open(
                f"smolgpt_{datetime.datetime.now().isoformat()}.pkl", "wb"
            ) as f:
                pickle.dump(model, f)
            raise e
