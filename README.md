# Tricycle
> A (very) minimal deep learning framework

Ever wanted to learn how a deep learning framework *actually* works under the hood? Tricycle might be for you.
Want to do anything else? Check out [pytorch](https://pytorch.org/)

## Overview
Tricycle is a minimal framework for deep learning. The goal of this library is 
not to build anything useful, but instead to get a good understanding of how 
deep learning works at every level. It is built using nothing but standard 
Python and Numpy which means that everything from automatic differentiation
to loss functions should (theoretically) be understandable to anyone who knows 
Python and Numpy.

Here are some things you can do with Tricycle:
- Create a tensor object
- Simple operations (addition, exponentiation, cosine, ...) on tensors
- Automatic differentiation of tensors
- Manipulate tensors with [einstein notation](https://en.wikipedia.org/wiki/Einstein_notation)
- Successfully train a simple neural network

Here are some things you can't do with Tricycle (yet):
- Use loss function that is not mean squared error
- Do anything efficiently
- Use any built in layers, optimisers, regularisation techniques etc
- Use a GPU
- Do anything actually useful

If you want to do these things, you should check out [pytorch](https://pytorch.org/)

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

