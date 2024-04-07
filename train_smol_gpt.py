"""

Training script for training a SmolGPT model on the complete
works of shakespeare.

The hyperparams for this model are very much a work in progress
"""

import os
import pickle
from copy import copy
from warnings import warn

import cupy
import mlflow
import pandas as pd
from tqdm import tqdm

from inference import generate
from tricycle.configs import SmolGPTConfig
from tricycle.dataset import CausalLMDataset
from tricycle.loss import cross_entropy
from tricycle.models import GPT
from tricycle.optimisers import StochasticGradientDescent
from tricycle_datasets.shakespeare import Shakespeare

config = SmolGPTConfig()
model = GPT(config)
model.display()

shakespeare = Shakespeare(vocab_size=config.vocab_size)
dataset = (
    CausalLMDataset(
        tokens=shakespeare.tokens,
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


def lr_schedule(
    step,
    max_learning_rate=config.max_learning_rate,
    min_learning_rate=config.min_learning_rate,
    warmup_steps=config.warmup_steps,
    total_steps=config.steps,
):
    """
    Linear decay LR schedule with warmup
    """
    # avoid an off by one error
    step += 1

    if warmup_steps:
        if total_steps < warmup_steps:
            raise ValueError(
                "Cannot have a warmup longer than the total number of steps"
            )
        if step < warmup_steps:
            return (step / warmup_steps) * max_learning_rate

    coef = (step - warmup_steps) / total_steps
    coef *= max_learning_rate - min_learning_rate
    return min_learning_rate + coef


with cupy.Device(1):
    model.to_gpu()

    def get_sample(sample_text, n_samples=50):
        sampled = []
        for i, next_token in tqdm(
            enumerate(generate(sample_text, model, shakespeare.tokeniser)),
            desc="evaulating",
            total=n_samples,
        ):
            if i > n_samples:
                break
            sampled.append(next_token)
        return shakespeare.tokeniser.decode(sampled)

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("SmolGPT:base")
    os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"

    best_loss = float("inf")
    losses = []
    n_steps = 25_000
    for step, (inputs, outputs) in tqdm(enumerate(dataset), total=n_steps):
        inputs = inputs.to_gpu()
        outputs = outputs.to_gpu()

        logits = model(inputs)
        loss = loss_fn(outputs, logits).from_vector().mean().mean()
        loss.backward()
        model.update(optimiser)

        # clean up the computational graph
        loss.cleanup()

        # step the learning rate
        optimiser.learning_rate = lr_schedule(step)

        if loss.numpy() > 1000:
            warn(f"Loss was {loss.numpy()} at step {step} - ")

        # log the loss
        losses.append(loss.numpy())
        mlflow.log_metric("loss", float(loss.numpy()), step=step)

        # occasionally log some metrics
        if step % 25 == 0:
            sample_text = "To be or not to be"
            predicted = get_sample(sample_text)
            mlflow.log_text(predicted, f"generated/{step}.txt")

        # checkpoint
        if loss < best_loss:
            with open("model.pkl", "wb") as f:
                pickle.dump(model, f)
            best_loss = loss

        if step >= n_steps:
            break

        step += 1
