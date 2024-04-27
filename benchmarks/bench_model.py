import numpy as np

from tricycle.configs import SmolGPTConfig
from tricycle.dataset import CausalLMDataset
from tricycle.layers import RMSNormV2
from tricycle.loss import cross_entropy
from tricycle.models import GPT, GPTV2
from tricycle.optimisers import StochasticGradientDescent
from tricycle_datasets.shakespeare import ShakespeareChar

np.random.seed(0)
config = SmolGPTConfig()
config.batch_size = 16
config.activation_fn = RMSNormV2()
# config.n_layers = 2
#
# config.n_heads = 6

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
    learning_rate=1e-3,
    weight_decay=0,
    momentum=0,
)


def train_improved_model():
    model = GPTV2(config)
    model.to_gpu(0)
    n_steps = 2
    for step, (inputs, outputs) in enumerate(dataset):
        logits = model(inputs)
        inputs.to_gpu(0)
        outputs.to_gpu(0)
        loss = loss_fn(outputs, logits).from_vector().mean().mean()
        loss.backward(clip=1)
        import pickle

        with open("model.pkl", "wb") as f:
            pickle.dump(model, f)
        exit()
        model.update(optimiser)

        # clean up the computational graph
        loss.cleanup()

        if step >= n_steps:
            break

        step += 1


def train_original_model():
    model = GPT(config)
    model.to_gpu(1)
    n_steps = 2
    for step, (inputs, outputs) in enumerate(dataset):
        inputs.to_gpu(1)
        outputs.to_gpu(1)
        logits = model(inputs)
        loss = loss_fn(outputs, logits).from_vector().mean().mean()
        loss.backward()
        model.update(optimiser)

        # clean up the computational graph
        loss.cleanup()

        if step >= n_steps:
            break

        step += 1


def train_original_model_gpu_0():
    model = GPT(config)
    model.to_gpu(0)
    n_steps = 2
    for step, (inputs, outputs) in enumerate(dataset):
        inputs.to_gpu(0)
        outputs.to_gpu(0)
        logits = model(inputs)
        loss = loss_fn(outputs, logits).from_vector().mean().mean()
        loss.backward()
        model.update(optimiser)

        # clean up the computational graph
        loss.cleanup()

        if step >= n_steps:
            break

        step += 1


__benchmarks__ = [
    (train_improved_model, train_improved_model, "Optimised multiple blocks")
    # (train_original_model, train_original_model_gpu_0, "Switched to GPU 0")
]
