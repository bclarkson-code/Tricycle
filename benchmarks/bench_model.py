import numpy as np

from tricycle.configs import SmolGPTConfig
from tricycle.dataset import CausalLMDataset
from tricycle.loss import cross_entropy
from tricycle.models import GPT, GPTV2
from tricycle.optimisers import StochasticGradientDescent
from tricycle_datasets.shakespeare import Shakespeare

np.random.seed(0)
config = SmolGPTConfig()
config.batch_size = 2
config.n_layers = 1

config.n_heads = 2

shakespeare = Shakespeare(vocab_size=config.vocab_size)
dataset = (
    CausalLMDataset(
        tokens=shakespeare,
        vocab_size=config.vocab_size,
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
    learning_rate=1e-3,
    weight_decay=0,
    momentum=0,
)


def train_improved_model():
    model = GPTV2(config)
    n_steps = 2
    for step, (inputs, outputs) in enumerate(dataset):
        logits = model(inputs)
        loss = loss_fn(outputs, logits).from_vector().mean().mean()
        loss.backward()
        model.update(optimiser)

        # clean up the computational graph
        loss.cleanup()

        if step >= n_steps:
            break

        step += 1


def train_original_model():
    model = GPT(config)
    n_steps = 2
    for step, (inputs, outputs) in enumerate(dataset):
        logits = model(inputs)
        loss = loss_fn(outputs, logits).from_vector().mean().mean()
        loss.backward()
        model.update(optimiser)

        # clean up the computational graph
        loss.cleanup()

        if step >= n_steps:
            break

        step += 1


# __benchmarks__ = [
#     (train_improved_model, train_original_model, "Use improved embedding")
# ]
