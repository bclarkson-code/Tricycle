"""

Training script for training a SmolGPT model on the complete
works of shakespeare.

The hyperparams for this model are very much a work in progress
"""

import os
import pickle
import uuid
from pathlib import Path

from tricycle import CUPY_ENABLED

if CUPY_ENABLED:
    import cupy as xp
else:
    import numpy as xp

import mlflow
from tqdm import tqdm

from inference import get_sample
from tricycle.configs import SmolGPTConfig
from tricycle.dataset import CausalLMDataset
from tricycle.loss import CrossEntropy
from tricycle.models import GPT
from tricycle.optimisers import AdamW
from tricycle.scheduler import lr_schedule
from tricycle_datasets.shakespeare import Shakespeare

xp.random.seed(0)
config = SmolGPTConfig()
model = GPT(config)
model.display()


dataset = Shakespeare(config.vocab_size)
dataloader = (
    CausalLMDataset(
        tokens=dataset,
        vocab_size=dataset.vocab_size,
        batch_size=config.batch_size,
        context_window=config.context_window,
        should_one_hot_encode=False,
    )
    .batch()
    .to_tensor()
    .to_vector()
    .shuffle()
)
loss_fn = CrossEntropy()
optimiser = AdamW(
    learning_rate=lr_schedule(
        0,
        max_learning_rate=config.max_learning_rate,
        min_learning_rate=config.min_learning_rate,
        warmup_steps=config.warmup_steps,
        total_steps=config.steps,
    ),
    weight_decay=config.weight_decay,
    betas=(config.beta1, config.beta2),
)


if CUPY_ENABLED:
    model.to_gpu(config.device_idx)


mlflow.set_tracking_uri(config.mlflow_tracking_uri)
mlflow.set_experiment("SmolGPT:tokeniser_1024:base")
os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"
unique_id = uuid.uuid4()

best_loss = xp.inf

losses = xp.zeros(config.steps)
for step in tqdm(range(config.steps), position=0):
    mlflow.log_params(config.dict())

    optimiser.step()
    batch_loss = 0
    # perform several forward and backward passes before doing a gradient
    # update to increase the effective batch size
    for _ in range(config.gradient_accumulation_steps):
        inputs, outputs = next(dataloader)
        if CUPY_ENABLED:
            inputs = inputs.to_gpu(config.device_idx)
            outputs = outputs.to_gpu(config.device_idx)

        # forward and backward pass
        logits = model(inputs)
        loss = loss_fn(outputs, logits).sum() / (
            config.gradient_accumulation_steps
            * config.batch_size
            * config.context_window
        )
        batch_loss += loss._data
        loss.backward()

    # Use the optimiser to update weights
    model.update(optimiser)

    mlflow.log_metric("loss", batch_loss, step=step)
    mlflow.log_metric("lr", float(optimiser.learning_rate), step=step)

    # step the learning rate
    optimiser.learning_rate = lr_schedule(
        step,
        max_learning_rate=config.max_learning_rate,
        min_learning_rate=config.min_learning_rate,
        warmup_steps=config.warmup_steps,
        total_steps=config.steps,
    )
    losses[step] = batch_loss

    if step % config.eval_interval == 0:
        # generate some text
        predicted = get_sample(model=model, dataset=dataset)
        mlflow.log_text(predicted, f"generated/{step}.txt")

        # checkpoint if new model better than old
        avg_loss = xp.mean(losses[step - config.eval_interval : step])
        if avg_loss < best_loss:
            Path("models").mkdir(exist_ok=True)
            with open(f"models/model_{unique_id}.pkl", "wb") as f:
                pickle.dump(model, f)
            best_loss = avg_loss
