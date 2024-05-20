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

from inference import generate
from tricycle.configs import SmolGPTConfig
from tricycle.dataset import CausalLMDataset
from tricycle.loss import BinaryCrossEntropy
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
loss_fn = BinaryCrossEntropy()
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


def get_sample(sample_text: str | None = None, n_samples: int = 50) -> str:
    """
    Given a prompt, generate some new tokens and return them as a string
    """
    if sample_text is None:
        # we need a full context window before we start generating so this
        # text is 256 characters long
        sample_text = """ROMEO:
He jests at scars that never felt a wound.
But, soft! what light through yonder window breaks?
It is the east, and Juliet is the sun.
Arise, fair sun, and kill the envious moon,
Who is already sick and pale with grief,
That thou her maid art far more fair than she:
Be not her maid, since she is envious;
Her vestal livery is but sick and green
And none but fools do wear it; cast it off.
It is my lady, O, it is my love!
O, that she knew she were!
She speaks yet she says nothing: what of that?
Her eye discourses; I will answer it.
I am too bold, 'tis not to me she speaks:
Two of the fairest stars in all the heaven,
Having some business, do entreat her eyes
To twinkle in their spheres till they return.
What if her eyes were there, they in her head?
The brightness of her cheek would shame those stars,
As daylight doth a lamp; her eyes in heaven
Would through the airy region stream so bright
That birds would sing and think it were not night.
See, how she leans her cheek upon her hand!
O, that I were a glove upon that hand,
That I might touch that cheek!

JULIET:
Ay me!

ROMEO:
She speaks:
O, speak again, bright angel! for thou art
As glorious to this night, being o'er my head
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
        position=1,
        leave=False,
    ):
        if i > n_samples:
            break
        sampled.append(next_token)
    decoded = dataset.decode(sampled)
    if not isinstance(decoded, str):
        decoded = "".join([chr(i) for i in decoded])
    return decoded


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
    # clean up the computational graph
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
        # log the loss

        # generate some text
        predicted = get_sample()
        mlflow.log_text(predicted, f"generated/{step}.txt")

        # checkpoint
        avg_loss = xp.mean(losses[step-config.eval_interval:step])
        if avg_loss < best_loss:
            Path("models").mkdir(exist_ok=True)
            with open(f"models/model_{unique_id}.pkl", "wb") as f:
                pickle.dump(model, f)
            best_loss = avg_loss
