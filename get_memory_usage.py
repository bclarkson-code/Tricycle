"""

Training script for training a SmolGPT model on the complete
works of shakespeare.

The hyperparams for this model are very much a work in progress
"""

import os
import pickle
import uuid

import cupy
import humanize
import mlflow
import numpy as np
from tqdm import tqdm

from inference import generate
from tricycle.configs import SmolGPTConfig
from tricycle.dataset import CausalLMDataset
from tricycle.loss import cross_entropy
from tricycle.models import GPT
from tricycle.optimisers import AdamW
from tricycle.scheduler import lr_schedule
from tricycle.utils import log_gpu_memory
from tricycle_datasets.shakespeare import ShakespeareChar

np.random.seed(0)
config = SmolGPTConfig()
config.batch_size = 12
config.n_layers = 1
model = GPT(config)


log_gpu_memory("initialisation")


dataset = ShakespeareChar()
dataset.vocab_size = 65
dataloader = (
    CausalLMDataset(
        tokens=dataset,
        vocab_size=dataset.vocab_size,
        batch_size=config.batch_size,
        context_window=config.context_window,
    )
    .batch()
    .to_tensor()
    .to_vector()
    .shuffle()
)

loss_fn = cross_entropy
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


model.to_gpu(config.device_idx)
log_gpu_memory("init_model")


def get_sample(sample_text: str | None = None, n_samples: int = 50) -> str:
    """
    Given a prompt, generate some new tokens and return them as a string
    """
    if sample_text is None:
        # we need a full context window before we start generating so this
        # text is 256 characters long
        sample_text = """'er my head
As is a winged messenger of heaven
Unto the white-upturned wondering eyes
Of mortals that fall back to gaze on him
When he bestrides the lazy-pacing clouds
And sails upon the bosom of the air.

JULIET:
O Romeo, Romeo! wherefore art thou Romeo?
"""
    sampled = []
    for i, next_token in tqdm(
        enumerate(generate(sample_text, model, dataset)),
        desc="evaluating",
        total=n_samples,
    ):
        if i > n_samples:
            break
        sampled.append(next_token)
    decoded = dataset.decode(sampled)
    if not isinstance(decoded, str):
        decoded = "".join([chr(i) for i in decoded])
    return decoded


# mlflow.set_tracking_uri(config.mlflow_tracking_uri)
# mlflow.set_experiment("SmolGPT:character:base")
# os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"
# unique_id = uuid.uuid4()

best_loss = float("inf")
losses = []
for step in tqdm(range(1)):
    optimiser.step()
    log_gpu_memory("start_loop")
    batch_loss = 0

    # perform several forward and backward passes before doing a gradient
    # update to increase the effective batch size
    for i in range(config.gradient_accumulation_steps):
        inputs, outputs = next(dataloader)
        log_gpu_memory("start_loop")
        inputs = inputs.to_gpu(config.device_idx)
        outputs = outputs.to_gpu(config.device_idx)
        log_gpu_memory("load_data")

        # forward and backward pass
        logits = model(inputs)
        log_gpu_memory("forward_pass")
        loss = loss_fn(outputs, logits).from_vector().e("ab->") / (
            config.gradient_accumulation_steps
            * config.batch_size
            * config.context_window
        )
        log_gpu_memory("calculate_loss")
        batch_loss += float(loss.numpy())
        loss.backward()
        log_gpu_memory("backward_pass")

        # delete intermediate values we dont need any more
        # TODO: statically allocate objects to avoid this
        loss.cleanup()
        log_gpu_memory("cleanup")

    # Use the optimiser to update weights
    model.update(optimiser)
    log_gpu_memory("update")

    #
    #     # clean up the computational graph
    #     # step the learning rate
    #     optimiser.learning_rate = lr_schedule(
    #         step,
    #         max_learning_rate=config.max_learning_rate,
    #         min_learning_rate=config.min_learning_rate,
    #         warmup_steps=config.warmup_steps,
    #         total_steps=config.steps,
    #     )
    #
    #     # log the loss
    #     losses.append(batch_loss)
    #     mlflow.log_metric("loss", batch_loss, step=step)
    #     mlflow.log_metric("lr", float(optimiser.learning_rate), step=step)
    #
    #     if step % config.eval_interval == 0:
    #         # generate some text
    #         predicted = get_sample()
    #         mlflow.log_text(predicted, f"generated/{step}.txt")
    #
    #         # checkpoint
    #         avg_loss = np.mean(losses[-config.eval_interval :])
    #         if avg_loss < best_loss:
    #             with open(f"models/model_{unique_id}_{step}.pkl", "wb") as f:
    #                 pickle.dump(model, f)
    #             best_loss = avg_loss
    #
import pandas as pd

df = pd.read_csv("memory.log")
df["diff"] = df["used_bytes"] - df["used_bytes"].shift()
df["diff_human"] = df["diff"].apply(humanize.naturalsize)
print(df)
