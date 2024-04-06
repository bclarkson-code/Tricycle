"""

Training script for training a SmolGPT model on the complete
works of shakespeare.

The hyperparams for this model are very much a work in progress
"""

import os
import pickle
from warnings import warn

import mlflow
import numpy as np
from tqdm import tqdm

from tricycle.configs import SmolGPTConfig
from tricycle.dataset import CausalLMDataset
from tricycle.loss import cross_entropy
from tricycle.models import GPT
from tricycle.optimisers import StochasticGradientDescent
from tricycle_datasets.shakespeare import Shakespeare

config = SmolGPTConfig()
model = GPT(config)
model.display()

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

model.to_gpu()

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("SmolGPT:debug")
os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"

best_loss = float("inf")
losses = []
for step, (inputs, outputs) in tqdm(enumerate(dataset), total=len(dataset)):
    inputs = inputs.to_gpu()
    outputs = outputs.to_gpu()

    logits = model(inputs)
    loss = loss_fn(outputs, logits).from_vector().mean().mean()
    loss.backward()
    model.update(optimiser)

    # clean up the computational graph
    loss.cleanup()

    if loss.numpy() > 1000:
        warn(f"Loss was {loss.numpy()} at step {step} - ")
    losses.append(loss.numpy())
    mlflow.log_metric("loss", float(loss.numpy()), step=step)

    # save best model
    if step % 250 == 0:
        average_loss = np.mean(losses)
        mlflow.log_metric("average_loss", float(average_loss), step=step)
        if average_loss < best_loss:
            best_loss = average_loss

            # save results
            with open("shakespeare_model.pkl", "wb") as f:
                pickle.dump(model, f)
