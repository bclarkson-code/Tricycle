"""
Training script for training a SmolGPT model on the complete
works of shakespeare.

The hyperparams for this model are very much a work in progress
"""

from tqdm import tqdm

from tricycle.configs import SmolGPTConfig
from tricycle.dataset import CausalLMDataset
from tricycle.loss import cross_entropy
from tricycle.models import GPT
from tricycle.optimisers import StochasticGradientDescent
from tricycle_datasets.shakespeare import Shakespeare

config = SmolGPTConfig()
model = GPT(config)

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

for inputs, outputs in tqdm(dataset):
    logits = model(inputs)
    loss = loss_fn(outputs, logits).from_vector().mean().mean()
    loss.backward()
    model.update(optimiser)
    break
