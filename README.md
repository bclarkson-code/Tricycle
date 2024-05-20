# Tricycle
> It don't go fast but it do be goin'

Ever wanted to learn how a deep learning framework *actually* works under the hood? Tricycle might be for you.

## Overview
Tricycle is a minimal framework for deep learning. The goal of this library is
not to match the speed or complexity or Pytorch or Tensorflow, but instead to get a good understanding of how
deep learning actually works at every level: from automatic differentiation all the way up to modern Language Models. It is built using nothing but standard
Python and Numpy which means that everything be understandable to anyone who knows a bit of Python.

Here are some things you can do with Tricycle:
- Create tensors
- Perform operations (addition, exponentiation, cosine, ...) on tensors
- Automatic differentiation of tensors
- Manipulate tensors with [einstein notation](https://en.wikipedia.org/wiki/Einstein_notation)
- Successfully train deep learning models
- Use a GPU
- Train a Transformer to produce infinite shakespeare(!)

If you would like to learn more about the process of building tricycle, you can check out my [blog](http://bclarkson-code.com)

## Usage
Theoretically, as a fully functional deep learning library, you can build any modern Deep Learning model with Tricycle. For example, this is how you can train a small language model on the shakespeare dataset:

```python
import pickle

from tqdm import tqdm

from tricycle.configs import SmolGPTConfig
from tricycle.dataset import CausalLMDataset
from tricycle.loss import cross_entropy
from tricycle.models import GPT
from tricycle.optimisers import AdamW
from tricycle_datasets.shakespeare import ShakespeareChar

config = SmolGPTConfig()
model = GPT(config)

tokens = ShakespeareChar(vocab_size=config.vocab_size)
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
optimiser = AdamW(
    learning_rate=config.max_learning_rate,
    weight_decay=config.weight_decay,
    betas=(config.beta1, config.beta2),
)

model.to_gpu()

best_loss = float("inf")
losses = []
for step in tqdm(range(config.steps)):
    optimiser.step()
    inputs, outputs = next(dataset)
    inputs = inputs.to_gpu()
    outputs = outputs.to_gpu()

    logits = model(inputs)
    loss = loss_fn(outputs, logits).sum() / (
        config.gradient_accumulation_steps
        * config.batch_size
        * config.context_window
    )
    loss.backward()

    model.update(optimiser)

# save results
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
```

This will fetch the complete works of shakespeare, build it into a dataset, tokenise it, and train a simple GPT on it.

As you can see, it looks pretty similar to other frameworks like PyTorch. However, because Tricycle is much smaller and simpler, if you want to figure out how something works, you can dive into the code and get an answer in a few minutes instead of hours.

For a proper training script with all the bells and whistles (logging, gradient accumulation etc.) take a look at train_smol_gpt.py which will train a transformer to produce infinite shakespeare in ~35 minutes (on my machine, with an RTX 3090).


## Contact
Want to learn more / have a chat / work together?
You can send an email to: [bclarkson-code@proton.me](mailto:bclarkson-code@proton.me)
