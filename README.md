# Tricycle
> It don't go fast but it do be goin'

Ever wanted to learn how a deep learning framework *actually* works under the hood? Tricycle might be for you.

Want to do anything else? Check out [pytorch](https://pytorch.org/)

## Overview
Tricycle is a minimal framework for deep learning. The goal of this library is
not to build anything useful, but instead to get a good understanding of how
deep learning works at every level. It is built using nothing but standard
Python and Numpy which means that everything from automatic differentiation
to loss functions should (theoretically) be understandable to anyone who knows
a bit of Python

Here are some things you can do with Tricycle:
- Create a tensor object
- Perform operations (addition, exponentiation, cosine, ...) on tensors
- Automatic differentiation of tensors
- Manipulate tensors with [einstein notation](https://en.wikipedia.org/wiki/Einstein_notation)
- Successfully train deep learning models

Here are some things you can't do with Tricycle (yet):
- Do anything at the speed of pytorch
- Use a GPU

If you want to do these things, you should check out [pytorch](https://pytorch.org/)

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
Tricycle uses [poetry](https://python-poetry.org/) to manage dependencies. Assuming it is installed, you
can install Tricycle by running:
```bash
poetry install
```

## Tests
Tricycle is tested using [pytest](https://docs.pytest.org/en/latest/)
```bash
poetry run pytest
```

## Contact
Want to learn more / have a chat / work together?
You can send an email to: [bclarkson-code@proton.me](mailto:bclarkson-code@proton.me)
