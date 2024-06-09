"""
Training script for SmolGPT a very small coding assistant.

The training script is pretty generic. You can tune the parameters in by
modifying the config.

Currently, we train the model on codeparrot, a supposedly cleaned and
deduped dataset of python files from github
"""

import os
import pickle
import uuid
from pathlib import Path

from tricycle import CUPY_ENABLED
from tricycle.tensor import Op, Tensor
from tricycle.utils import optimal_n_tokens

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
from tricycle_datasets.codeparrot import CodeParrot

# fix the seed for reproducibility
xp.random.seed(0)
config = SmolGPTConfig()
model = GPT(config)

model.display()

# Use corrected Chinchilla scaling to estimate the compute-optimal number of
# tokens and steps we should train for
n_tokens, n_steps = optimal_n_tokens(model, config)


def load_datasets(
    n_tokens: int, config: SmolGPTConfig
) -> tuple[CodeParrot, CodeParrot, CausalLMDataset, CausalLMDataset]:
    """
    Load tokens, batch and shuffle them.
    """

    # if you are loading this for the first time, this can take a while.
    # it will create some big cache files in ~/.cache/huggingface that you might
    # want to clean up once you are done with the dataset
    print("Loading dataset")
    train_dataset = CodeParrot(config.vocab_size, split="train")

    # trim the training dataset to the chinchilla optimal number of tokens
    train_dataset.tokens = train_dataset.tokens[:n_tokens]
    valid_dataset = CodeParrot(config.vocab_size, split="valid")

    print("Loading dataloaders")
    train_dataloader = (
        CausalLMDataset(
            tokens=train_dataset.tokens,
            vocab_size=train_dataset.vocab_size,
            batch_size=config.batch_size,
            context_window=config.context_window,
        )
        .batch()
        .shuffle()  # only shuffle train dataset.
        .to_tensor()
    )
    valid_dataloader = (
        CausalLMDataset(
            tokens=valid_dataset.tokens,
            vocab_size=valid_dataset.vocab_size,
            batch_size=config.batch_size,
            context_window=config.context_window,
        )
        .batch()
        .to_tensor()
    )
    return (
        train_dataset,
        valid_dataset,
        train_dataloader,
        valid_dataloader,
    )


def estimate_loss(
    model: GPT,
    valid_dataloader: CausalLMDataset,
    config: SmolGPTConfig,
    loss_fn: Op,
) -> float:
    batch_loss = 0
    for valid_step, (inputs, outputs) in tqdm(
        enumerate(valid_dataloader), total=config.eval_steps, desc="Validation"
    ):
        if valid_step == config.eval_steps:
            break

        assert isinstance(inputs, Tensor)
        assert isinstance(outputs, Tensor)
        if CUPY_ENABLED:
            inputs = inputs.to_gpu(config.device_idx)
            outputs = outputs.to_gpu(config.device_idx)

        # forward and backward pass
        logits = model(inputs)
        loss = loss_fn(outputs, logits)
        batch_loss += loss.array / config.eval_steps

    return batch_loss


train_dataset, valid_dataset, train_dataloader, valid_dataloader = (
    load_datasets(n_tokens, config)
)
loss_fn = CrossEntropy()
optimiser = AdamW(
    learning_rate=lr_schedule(
        0,
        max_learning_rate=config.max_learning_rate,
        min_learning_rate=config.min_learning_rate,
        warmup_steps=config.warmup_steps,
        total_steps=n_steps,
    ),
    weight_decay=config.weight_decay,
    betas=(config.beta1, config.beta2),
)


if CUPY_ENABLED:
    model.to_gpu(config.device_idx)


mlflow.set_tracking_uri(config.mlflow_tracking_uri)
mlflow.set_experiment("SmolGPT:codeparrot:base")
os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"
with mlflow.start_run() as run:
    unique_id = uuid.uuid4()

    best_loss = xp.inf

    losses = xp.zeros(n_steps)
    for step in tqdm(range(n_steps), position=0):
        mlflow.log_params(config.dict())

        optimiser.step()
        batch_loss = 0

        # perform several forward and backward passes before doing a gradient
        # update to increase the effective batch size
        for _ in range(config.gradient_accumulation_steps):
            inputs, outputs = next(train_dataloader)
            assert isinstance(inputs, Tensor)
            assert isinstance(outputs, Tensor)
            if CUPY_ENABLED:
                inputs = inputs.to_gpu(config.device_idx)
                outputs = outputs.to_gpu(config.device_idx)

            # forward and backward pass
            logits = model(inputs)
            loss = loss_fn(outputs, logits)
            batch_loss += loss.array / config.gradient_accumulation_steps
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
            total_steps=n_steps,
        )
        losses[step] = batch_loss

        if step % config.eval_interval == 0:
            # generate some text
            predicted = get_sample(
                model=model,
                tokeniser=valid_dataset.tokeniser,
                sample_tokens=valid_dataset.tokens[: config.context_window],
            )
            mlflow.log_text(predicted, f"generated/{step}.txt")

            # esimate validation loss
            valid_loss = estimate_loss(
                model=model,
                valid_dataloader=valid_dataloader,
                config=config,
                loss_fn=loss_fn,
            )
            mlflow.log_metric("valid_loss", valid_loss, step=step)

            # checkpoint if new model better than old
            if valid_loss < best_loss:
                Path("models").mkdir(exist_ok=True)
                with open(f"models/model_{unique_id}.pkl", "wb") as f:
                    pickle.dump(model, f)
                best_loss = valid_loss
