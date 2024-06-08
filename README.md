# Tricycle
Tricycle is a fast, minimal, fully funtional deep learning library written from scratch using only python an numpy.
The file 'train_smol_gpy.py' trains a 49M, GPT-2 style model to produce python code in ~2 days on my RTX 3090.

The entire library, from the automatic differentiation engine to the transformer block, is written in ~4500 lines of python + numpy code.
Using [CuPY](https://cupy.dev/), all Tricycle code can run on a GPU and is only about ~TODO: insert comparision to pytorch here.~ % [slower than pytorch](#comparison-with-pytorch).




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

If you would like to learn more about the process of building tricycle, you can check out my [blog](http://bclarkson-code.com)

## Usage
Theoretically, as a fully functional deep learning library, you can build any modern Deep Learning model with Tricycle. For example, this is how you can train a simple neural network on the iris dataset:

```python
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris

from tricycle.activation import ReLU
from tricycle.dataset import InfiniteBatchDataset
from tricycle.layers import Dense, Sequential
from tricycle.loss import cross_entropy
from tricycle.optimisers import StochasticGradientDescent

BATCH_SIZE = 64
LEARNING_RATE = 3e-2
N_STEPS = 100

# load iris data from sklearn
X, y = load_iris(return_X_y=True)

# one hot encode y
y = np.eye(3)[y.astype(int)]

# create a dataset
ds = InfiniteBatchDataset(X, y, batch_size=BATCH_SIZE)
batches = ds.to_tensor().to_vector()

# create a model
layer_1 = Dense(4, 16)
relu = ReLU()
layer_2 = Dense(16, 3)
model = Sequential(layer_1, relu, layer_2)

# create a loss function and an optimiser
loss_fn = cross_entropy
optimiser = StochasticGradientDescent(learning_rate=LEARNING_RATE)

losses = []
for step, (inputs, outputs) in enumerate(batches):
    if step > N_STEPS:
        break

    y_pred = model(inputs)
    loss = loss_fn(outputs, y_pred).from_vector().e("a->") / BATCH_SIZE
    loss.backward()
    losses.append(loss)

    model.update(optimiser)
    model.zero_grad()

# Plot a graph of the loss
plt.plot(losses)
plt.show()
```

As you can see, it is about as complex as any other deep learning framework.

## Installation
Tricycle uses [conda](https://docs.conda.io/en/latest/) to manage dependencies. While we do support CPU-only computation, at time of writing, not effort has been put into optimising it. If you do have a CUDA capable GPU I would strongly reccommend installing the gpu version of Tricycle.

Training Smol GPT on my GPU takes ~30 mins while training Smol GPT on CPU takes ~122 hours.

### GPU Installation
If you have a CUDA capable GPU, you can install Tricycle as follows.
```bash
conda env create -f environment.yml
conda activate tricycle
```
If you want to install test-dependencies you can do the following.

```bash
conda env create -f environment.test.yml
conda activate tricycle
```

### CPU Installation
If you want to install Tricycle for CPU, you can do the following.
```bash
conda env create -f environment.cpu_only.yml
conda activate tricycle
```

If you want to install test-dependencies you can do the following.
```bash
conda env create -f environment.cpu_only.test.yml
conda activate tricycle
```

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

For a proper training script with all the bells and whistles (logging, gradient accumulation etc.) take a look at `train_smol_gpt.py` which will train a transformer to produce infinite shakespeare in ~35 minutes (on my machine, with an RTX 3090).


## Contact
Want to learn more / have a chat / work together?
You can send an email to: [bclarkson-code@proton.me](mailto:bclarkson-code@proton.me)
