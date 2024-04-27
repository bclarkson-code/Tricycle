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

from inference import generate
from tricycle.configs import SmolGPTConfig
from tricycle.dataset import CausalLMDataset
from tricycle.loss import cross_entropy
from tricycle.models import GPTV2
from tricycle.optimisers import StochasticGradientDescent
from tricycle.scheduler import lr_schedule
from tricycle_datasets.shakespeare import ShakespeareChar

np.random.seed(0)
config = SmolGPTConfig()
model = GPTV2(config)
model.display()


shakespeare = ShakespeareChar()
shakespeare.vocab_size = 65
dataset = (
    CausalLMDataset(
        tokens=shakespeare,
        vocab_size=shakespeare.vocab_size,
        batch_size=config.batch_size,
        context_window=config.context_window,
    )
    .batch()
    .to_tensor()
    .to_vector()
    .shuffle()
)
loss_fn = cross_entropy
optimiser = StochasticGradientDescent(
    learning_rate=lr_schedule(
        0,
        max_learning_rate=config.max_learning_rate,
        min_learning_rate=config.min_learning_rate,
        warmup_steps=config.warmup_steps,
        total_steps=config.steps,
    ),
    weight_decay=config.weight_decay,
    momentum=config.momentum,
)


model.to_gpu()


def get_sample(sample_text, n_samples=50):
    sampled = []
    for i, next_token in tqdm(
        enumerate(generate(sample_text, model, shakespeare)),
        desc="evaulating",
        total=n_samples,
    ):
        if i > n_samples:
            break
        sampled.append(next_token)
    decoded = shakespeare.decode(sampled)
    if not isinstance(decoded, str):
        decoded = "".join([chr(i) for i in decoded])
    return decoded


mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("SmolGPT:character:debug")
os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"

best_loss = float("inf")
losses = []
for step, (inputs, outputs) in tqdm(enumerate(dataset), total=config.steps):
    inputs = inputs.to_gpu()
    outputs = outputs.to_gpu()

    logits = model(inputs)
    loss = loss_fn(outputs, logits).from_vector().mean().mean()
    loss.backward()
    model.update(optimiser)

    # clean up the computational graph
    loss.cleanup()

    # step the learning rate
    optimiser.learning_rate = lr_schedule(
        step,
        max_learning_rate=config.max_learning_rate,
        min_learning_rate=config.min_learning_rate,
        warmup_steps=config.warmup_steps,
        total_steps=config.steps,
    )

    if loss.numpy() > 1000:
        warn(f"Loss was {loss.numpy()} at step {step} - ")

    # log the loss
    losses.append(loss.numpy())
    mlflow.log_metric("loss", float(loss.numpy()), step=step)
    mlflow.log_metric("lr", float(optimiser.learning_rate), step=step)

    # occasionally try generating some text
    if step % 50 == 0:
        sample_text = "To be or not to be"
        predicted = get_sample(sample_text)
        mlflow.log_text(predicted, f"generated/{step}.txt")

    # checkpoint
    if loss < best_loss:
        with open("model.pkl", "wb") as f:
            pickle.dump(model, f)
        best_loss = loss

    if step >= config.steps:
        break

    step += 1
