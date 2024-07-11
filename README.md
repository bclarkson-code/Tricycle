# Tricycle

<p align="center">
    <img width="223" alt="tricycle_logo" src="https://github.com/bclarkson-code/Tricycle/assets/57139598/62405944-b27b-49bc-93c3-17ba93fc8ad7">
</p>

Tricycle is a fast, minimal, fully functional deep learning library written from scratch using only python and numpy.

While I've tried to make it easy to follow, Tricycle is not just an educational toy: the file `train_smol_gpt.py` trains GPT-2 (124M) on 2.5B (chinchilla optimal) tokens in just under 3 days on my GPU (RTX 3090).


The entire library, from the automatic differentiation engine to a GPT, should be understandable to anyone with a bit of python experience and I encourage you to explore the codebase.

The entire library, from the automatic differentiation engine to a GPT, should be understandable to anyone with a bit of python experience.

All Tricycle code can run on either a CUDA-capable GPU or a CPU (although optimisation efforts have been focussed on the GPU so CPU computation is really slow).

## Table of Contents
- [Tricycle](#tricycle)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
    - [CPU Installation](#cpu-installation)
  - [Training a GPT on Shakespeare](#training-a-gpt-on-shakespeare)
  - [How it works](#how-it-works)
    - [Automatic Differentiation](#automatic-differentiation)
    - [Einsum](#einsum)
      - [Summing along an axis](#summing-along-an-axis)
      - [Sum of an entire tensor](#sum-of-an-entire-tensor)
      - [Transpose](#transpose)
      - [Matrix multiplication](#matrix-multiplication)
    - [Building a simple neural network](#building-a-simple-neural-network)
    - [Optimisations](#optimisations)
      - [Batching](#batching)
      - [GPU](#gpu)
      - [Fusing](#fusing)
      - [Other optimisations](#other-optimisations)
        - [Inplace tensor updates](#inplace-tensor-updates)
        - [Mathematical optimisations](#mathematical-optimisations)
        - [Hardware optimisations](#hardware-optimisations)
    - [Building a Language model](#building-a-language-model)
      - [Input block](#input-block)
      - [Transformer Block](#transformer-block)
      - [Attention Block](#attention-block)
      - [MLP Block](#mlp-block)
      - [Output](#output)
  - [What's Next?](#whats-next)
  - [Contact](#contact)

## Installation

Tricycle uses [conda](https://docs.conda.io/en/latest/) to manage dependencies. While we do support CPU-only computation, optimisation efforts have been focussed on GPU computation so it is pretty slow. If you do have a CUDA capable GPU I would strongly recommend installing the gpu version of Tricycle.

If you have a CUDA capable GPU, you can install Tricycle as follows.

```bash
conda env create -f requirements/environment.yml -n tricycle
conda activate tricycle
```

<details>
    <summary>Test installation</summary>
If you want to install test dependencies with GPU support you can do the following.

```bash
conda env create -f requirements/environment.test.yml -n tricycle
conda activate tricycle
```

If you want to install test dependencies on CPU you can do the following.

```bash
conda env create -f requirements/environment.cpu.test.yml -n tricycle
conda activate tricycle
```

</details>

## Training a GPT on Shakespeare

The following toy script will train a small GPT to generate convincing Shakespeare.
On my RTX 3090, this takes ~9 mins. For a more realistic training script with metric tracking, gradient accumulation, a validation dataset etc, take a look at `train_smol_gpt.py`

I've chosen some sensible default values for this model in `src/tricycle/configs.py:ShakespeareConfig`. Feel free to play around with these and see what happens.
If you are running out of GPU memory, try dropping the batch size and if your GPU is slow, try reducing the number of steps.

If you don't have a CUDA capable GPU, you can run the script on CPU but it will take a while. You'll probably want to try dropping the number of steps and leave this running overnight.
```python
import pickle

from tqdm import tqdm

from tricycle import GPU_ENABLED
from tricycle.configs import ShakespeareConfig
from tricycle.dataset import CausalLMDataset
from tricycle.loss import CrossEntropy
from tricycle.models import GPT
from tricycle.optimisers import AdamW
from tricycle_datasets.shakespeare import Shakespeare

config = ShakespeareConfig()
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
    .shuffle()
    .to_tensor()
)
loss_fn = CrossEntropy()
optimiser = AdamW(
    learning_rate=config.max_learning_rate,
    weight_decay=config.weight_decay,
    betas=(config.beta1, config.beta2),
)

if GPU_ENABLED:
    dataset = dataset.to_gpu()
    model.to_gpu()

loading_bar = tqdm(range(config.steps))
for step in loading_bar:
    optimiser.step()
    inputs, outputs = next(dataset)

    logits = model(inputs)
    loss = loss_fn(outputs, logits)
    loss.backward()

    loading_bar.set_description(f"loss: {loss.numpy().item():.3f}")
    model.update(optimiser)

# save results
with open("model.pkl", "wb") as f:
    if GPU_ENABLED:
        model.from_gpu()
    pickle.dump(model, f)
```

Once trained, you can generate infinite shakespeare plays as follows:

```bash
python inference.py model.pkl
```

## How it works

Tricycle code centers around objects called `Tensors`. A `Tensor` is a wrapper around a Numpy array that adds some extra features:

```python
from tricycle.tensor import to_tensor

tensor = to_tensor([1,2,3])
print(tensor) # Output: Tensor([1. 2. 3.])
```

You can do a lot of things with a tensor

```python
from tricycle.functions import Softmax

a = to_tensor([1,2,3])
b = to_tensor([4,5,6])

# addition
print(a + b) # Output: Tensor([5. 7. 9.], name=badd)

# comparison
print(a < b) # Output: Tensor([ True  True  True])

# more complex functions
print(Softmax()(a)) # Output: Tensor([0.09003057 0.24472848 0.66524094], name=softmax)

```

### Automatic Differentiation

Unlike vanilla Numpy, every operation in Tricycle is attached to a derivative.
When you do some operations on your `Tensor`, Tricycle keeps track of what 
you did and allows you to differentiate the output.

```python
x = to_tensor(2)

y = x ** 2 + 3 * x + 4
print(y) # Output: Tensor(14.0, name=+ 4)

# derivative of y with respect to (wrt) x is
# 2 * x + 3 = 7
y.backward() # differentiate wrt y
print(x.grad) # Output: Tensor(7.0)
```

This works on multidimensional tensors

```python
import numpy as np

shape = (6,5,4,3,2)
a = to_tensor(np.random.random(shape))
b = to_tensor(np.random.random(shape))

c = a * b # elementwise multiply

c.backward() # differentiate wrt c
assert a.grad.close_to(b) # derivative of c wrt a is b
assert b.grad.close_to(a) # derivative of c wrt b is a
```

And even works through complex operations like attention

```python
from tricycle.blocks import MultiHeadSelfAttention

attention = MultiHeadSelfAttention(
    embedding_dim=32,
    n_heads=2,
    context_window=32,
)

# batch_size, n_tokens, embedding_dim
shape = (4,32,32)
input = to_tensor(np.ones(shape), is_batched=True)

output = attention(input)
output.backward() # differentiate wrt output

print(input.grad) # Output: Tensor([[[ 2.5441039  -2.0558214  -1.7923143  ...
assert input.grad.shape == (4,32,32)
```

When you run an operation (`Op`), the output has two pieces of information attached:

- `args`: The inputs to the function
- `back_fns`: The functions that should be executed to calculate the derivative wrt each of the inputs

Surprisingly, this all that you need to perform automatic differentiation on an arbitrarily complicated sequence of `Op`s.
Because we keep track of the `args` for each operation, we can start at the output of a set of `Op`s and traverse through them to reach every input to the sequence: the operations form a tree.

Thanks to the [chain rule](https://en.wikipedia.org/wiki/Chain_rule), if we apply each `back_fn` that we pass through on our way through the tree, when we get to an input, we will have calculated the derivative of the output wrt the input.
Despite implementing it myself, I still feel like this couldn't possibly work, and yet it does!

The entirety of the algorithm can be found in [`tensor.py`](https://github.com/bclarkson-code/Tricycle/blob/main/src/tricycle/tensor.py#L145).

It ends up being a topological sort to figure out which order to traverse the tree and then a simple traversal, applying the `back_fns` along the way.

There isn't enough space to explain this properly here so you can find a detailed explanation [here](https://bclarkson-code.com/posts/llm-from-scratch-scalar-autograd/post.html).

### Einsum

Tricycle makes use of (in my opinion underutilised) einsum operations.
Einsum is a generalisation of a large number of matrix operations.

You can use it by assigning each axis in your matrices a letter of the
alphabet (called an index). You can define the operation you want to perform
by simply listing the indices you want in your inputs and output, separated by
an arrow.

For example, you can define the transpose of a 2d tensor as follows:

```python
from tricycle.einsum import Einsum

a = to_tensor([[1,2],[3,4]])
print(Einsum("ij->ji")(a)) # Output: Tensor([[1. 3.], [2. 4.]], name=einsum ij->ji)
```

Here, we use einsum to swap indices i and j: a transpose.

There are only two rules to remember with einsum:

- If an index does not appear in the output, any inputs that contain it
  will be summed along that axis:

  ```python
  print(Einsum("ij->i")(a)) # Tensor([3. 7.], name=einsum ij->i)
  ```

- If an index appears in more than one input, the tensors will be multiplied
  along that axis

  ```python
  b = to_tensor([[5,6],[7,8])
  print(Einsum("ij,jk->ik")(a,b)) # Tensor([[19. 22.], [43. 50.]], name=einsum ij,jk->ik)
  ```

For example:

#### Summing along an axis

https://github.com/bclarkson-code/Tricycle/assets/57139598/c575c958-19ed-4406-8a1b-d2390663ba96

#### Sum of an entire tensor

https://github.com/bclarkson-code/Tricycle/assets/57139598/efbb5eaa-656c-40e5-a32d-b0f5e7bd28f5

#### Transpose

https://github.com/bclarkson-code/Tricycle/assets/57139598/f8b35a6b-f102-44f1-a7cd-b6b2e765f275

#### Matrix multiplication

https://github.com/bclarkson-code/Tricycle/assets/57139598/1ed18428-11de-4990-a0f4-12d1310d6898

Because every `Op` in Tricycle needs a derivative, we need to figure out what the
derivative of `Einsum` is. I was worried that this would be complex but
thankfully, if you sit down and go through the
maths (index notation is really helpful here) it turns out to be pretty simple:
Just follow these two rules:

- Swap the indices for the input and output
- Replace the original input with your current derivative

For example, the derivative of a transpose works like this:

```python
# forward operation
y = Einsum('ij->ji')(a)

# swap the input with the current grad (a grid of ones in this case)
grad = to_tensor(np.ones_like(y))

# swap the indices
derivative = Einsum('ji->ij')(grad)
```

And for a more complex operation (a dense layer on a 4d input) like this:

```python
# forward operation
input = to_tensor(np.random.random((5, 4, 3, 2)))
weights = to_tensor(np.random.random((3,6)))
y = Einsum('zxTb,bW->zxTW')(inputs, weights)

grad = to_tensor(np.ones_like(y))

# swap the indices + replace inputs
derivative = Einsum('zxTb,zxTW->bW')(inputs, grad)
```

This little trick significantly simplifies code, as well as reducing the
amount of maths I had to do to implement different operations.

### Building a simple neural network

Einsum and an automatic differentiation engine are all we need to build a simple neural network. Lets try to train a model on the [iris dataset](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html)
We can start with a [`Dense` Layer](https://github.com/bclarkson-code/Tricycle/blob/main/src/tricycle/layers.py#L34).

```python
from tricycle.layers import Dense

x = to_tensor([1,2,3])
layer = Dense(from_size=3, to_size=1)

print(layer(x)) # Output: Tensor([-2.238703], name=dense)
```

Next, neural networks need a non-linearity (otherwise they reduce to expensive linear regressions).

Tricycle has a few [non-linearities](https://github.com/bclarkson-code/Tricycle/blob/main/src/tricycle/activation.py) (also called activation functions). Here we can choose the simplest: `ReLU`.

```python
from tricycle.activation import ReLU

x = to_tensor([-1, 0, 1])
activation_fn = ReLU()

print(activation_fn(x)) # Output: Tensor([0. 0. 1.], name=> 0)
```

We also need a loss function. We're predicting a category so we can use CrossEntropy

```python
from tricycle.loss import CrossEntropy

label = to_tensor([0, 1, 2], dtype=int)
predicted = to_tensor([[0,0,1], [0,0,1], [0,0,1]])
loss = CrossEntropy()

print(loss(label, predicted)) # Output: Tensor(1.2181114, name=cross_entropy)
```

Finally, we need an optimiser to update our weights. We can use [Stochastic Gradient Descent](https://github.com/bclarkson-code/Tricycle/blob/main/src/tricycle/optimsers.py#L14).
In Tricycle, you can use an optimiser the weights of a model as follows:

```python
from tricycle.activation import ReLU
from tricycle.layers import Dense, Sequential
from tricycle.optimisers import StochasticGradientDescent

# build a model
layer_1 = Dense(4, 16)
layer_2 = Dense(16, 3)
relu = ReLU()
model = Sequential(layer_1, relu, layer_2)

# create an optimiser
optimiser = StochasticGradientDescent(learning_rate=1e-1)

# do a forward and backward pass
x = to_tensor([1,2,3,4])
out = model(x)
out.backward()

# update the weights
model.update(optimiser)
```

We can put all of this together to train a simple neural network on the iris
dataset.

```python
import numpy as np
from sklearn.datasets import load_iris

from tricycle.activation import ReLU
from tricycle.tensor import to_tensor
from tricycle.layers import Dense, Sequential
from tricycle.loss import CrossEntropy
from tricycle.optimisers import StochasticGradientDescent

LEARNING_RATE = 1e-1
N_STEPS = 1000

np.random.seed(42)
X, y = load_iris(return_X_y=True)
inputs = to_tensor(X, is_batched=True)

# The class labels need to be ints for cross entropy
outputs = to_tensor(y, is_batched=True, dtype=int)

# create a model
layer_1 = Dense(4, 16)
layer_2 = Dense(16, 3)
relu = ReLU()
model = Sequential(layer_1, relu, layer_2)

loss_fn = CrossEntropy()
optimiser = StochasticGradientDescent(learning_rate=LEARNING_RATE)

for step in range(N_STEPS):
    y_pred = model(inputs)
    loss = loss_fn(outputs, y_pred)
    if step == 0:
        print(f"Initial loss: {loss}") # Output: Initial loss: Tensor(3.974701, name=cross_entropy)

    loss.backward()
    model.update(optimiser)

print(f"Final loss: {loss}") # Output: Final loss: Tensor(0.08622341, name=cross_entropy)

# Calculate accuracy
predicted_labels = np.argmax(y_pred.array, axis=-1)
accuracy = (predicted_labels == outputs.array).mean()
print(f"Accuracy: {accuracy:.2f}") # Output: Accuracy: 0.97
```

### Optimisations

Deep learning is famously computationally heavy. If we want to train anything
in a reasonable amount of time, there are several optimisations we need to make.

#### Batching

The first, optimisation is batching. Instead of applying operations to each 
input individually, if we are clever about how we design an operation, we can
apply an operation to multiple operations at once.

For example, suppose we are multiplying a batch of tensors by a weight matrix.
We could do it like this:

```python
# batch of 1024 64x64 tensors
inputs = to_tensor(np.ones((1024, 64, 64)))
weights = to_tensor(np.random.random((64,64)))

output = [Einsum('ij,jk->ik')(inp, weights) for inp in inputs]
# 62.2 ms ± 186 μs per loop (mean ± std. dev. of 7 runs, 10 loops each)
```

But we can use the properties of `Einsum` to do the same thing like this

```python
output = Einsum('aij,jk->aik')(inputs, weights)
# 29.1 ms ± 99.2 μs per loop (mean ± std. dev. of 7 runs, 10 loops each)
```

Which is more than 2x faster.

Some `Op`s in tricycle behave slightly differently, depending on
whether a tensor batched or not. You can tell tricycle to use the batched
version of `Op`s for a tensor by simply calling `.to_batched`. To convert it
back, you can call `.from_batched`.

#### GPU

As well as batching, another improvement that has a big impact on performance
is using a GPU. For this, we can use a library called [CuPY](https://cupy.dev/).
CuPY lets you run Numpy code on a GPU. This means that we can use the same code
for CPU as well as GPU computation which greatly simplifies the codebase (
and avoids me needing to write CUDA kernels, for now).

Every tensor in tricycle has an `.xp` method. By default, this is just the
Numpy library:

```
import numpy as np

tensor = to_tensor([1,2,3])

assert tensor.xp == np
```

But if you call `.to_gpu` on a tensor, this is the Cupy library:

```
import cupy as cp

tensor = to_tensor([1,2,3])

tensor.to_gpu()

assert tensor.xp == cp
```

(`xp` stands for `np` or `cp` because x is an "unknown"). This is really handy
because it lets us write device-agnostic functions like this:

```python
def forward(self, tensor: Tensor):
    """
    Apply softmax. The softmax is only applied to the final
    dimension of the tensor
    Note: the tensor is normalised for numeric stability
    """
    xp = tensor.xp

    exp = xp.exp(
        # subtract the largest value for numeric stability
        tensor.array
        - xp.max(tensor.array, axis=-1, keepdims=True)
    )
    denominator = xp.sum(exp, axis=-1, keepdims=True)
    self._out = exp / denominator

    return Tensor(
        self._out,
        args=(tensor,),
        name="softmax",
        is_batched=tensor.is_batched,
        back_fns=(self.back_fn,),
    )
```

Because Cupy has the same interface as Numpy, this function will automatically
run on the right device, with no code changes.

#### Fusing

One of the problems I faced when trying to use Tricycle is that it used up
a lot more memory than I expected. Because the `args` and `back_fns` need to
be stored for every `Op`, a lot of memory was being used to store intermediate
values.

For more complex operations like `Softmax`, this quickly adds up. However,
we can avoid a lot of this overhead by pre-computing the combined derivative.
In the case of `Softmax` (see above), we could have built it entirely out of
low level Tricycle operations and this does work. When you sit down and work
out the derivative for softmax manually, it turns out to be pretty simple:

```python
def backward(self, grad: Tensor) -> Tensor:
    xp = grad.xp

    inner = xp.sum(grad.array * self._out, axis=-1, keepdims=True)
    self._grad = self._out * (grad.array - inner)
    return to_tensor(
        self._grad,
        is_batched=grad.is_batched,
        requires_grad=grad.requires_grad,
    )
```

This kind of operation is a very common optimisation technique in deep learning
called 'Operator Fusing'. This ends up being a big optimisation for tricycle
because it lets us replace operations like `MultiHeadSelfAttention`, which
would usually have 10s of intermediate values, with a single `forward` and
`backward` function with a minimal set of intermediate values.

#### Other optimisations

While batching, using a GPU and fusing are the major optimisations, I'd like
to provide some honourable mentions.

##### Inplace tensor updates

While probably obvious to many readers, updating tensors in-place rather than
replacing them with a new tensor caused a big speed up.

##### Mathematical optimisations

Operations like `CrossEntropy` can be implemented by applying a softmax and then
applying the cross entropy operation but, if you do a bit of algebra,
you can do something called the `log-sum-exp` trick to simplify the expression
and cut down on the computations needed.

##### Hardware optimisations

As mentioned above, the GPU computation was performed on an NVIDIA RTX 3090.
Understandably, this gets quite hot when training (probably something to do with
it being in my cupboard?) which can reduce performance due to thermal
throttling. However, I found that by removing my computer case and placing
a household fan on top, I get about 30% better performance.

![IMG_0713](https://github.com/bclarkson-code/Tricycle/assets/57139598/958f12b4-caaa-4f2a-b9d0-2f5a7fc1e5a5)

Putting all of these things together, Tricycle can train a small language model on shakespeare in ~30 mins. Andrej Karpathy can [do this in pytorch](https://github.com/karpathy/nanoGPT/tree/master) in around 7 minutes on my machine (with a like-for-like config) which, given that the entire Tricycle project is in python, means that Tricycle is surprisingly fast. That said, more work is needed to get the speed up.

### Building a Language model

Now that we've got an automatic differentiation engine, we can start actually
doing things with it. GPT 2 was arguably the first to use the modern stack for
language generation. Even modern state of the art models like llama3 use the
same basic architecture and training methods, with only a few small tweaks
(e.g swapping layer norm with rms norm). Because I don't have access to many
GPUs, we'll be training a smaller (49M parameter) version.

To build our GPT, we first need to understand its architecture:
![GPT](https://github.com/bclarkson-code/Tricycle/assets/57139598/14b16802-2bfd-4d10-99b9-168e5cc6290e)

There are a few important things to note in this diagram. First, the
transformer is built out of 3 main pieces, the input block, a stack of
transformer blocks and then an output layer. The input layer turns a list of
tokens into a list of embeddings (each token gets projected to an embedding
vector). The stack of transformer blocks process the embeddings, but leave
their shape untouched and then the output layer converts each embedding into a
vector that is the same length as the number of tokens in our vocabulary (more
on this later).

This means that the transformer accepts a fixed number of tokens and predicts
a fixed number of tokens. The number of tokens it accepts is usually called
the context window but is sometimes called the block size or sequence length.

Also, it means that we can make our transformer bigger or smaller pretty easily
by simply increasing the number of tokens in our context window, the size of
our embeddings and the number of transformer blocks in our stack. (There is
also the number of transformer heads but more on this later too).

#### Input block

We know the input block needs to take a list of tokens as an input and return
a list of embeddings. We can do this with a dense layer. We can one-hot
encode a token into a vector of 0s with a single 1 corresponding to the
token id (e.g `2 -> [0,0,1,0,...,0]`). Then we can pass this through a dense
layer to convert it from a `1 x vocab_size` vector to a `1 x embedding_size`
vector.

However, this is a very expensive operation. For each token, we need to
do a multiplication by a `vocab_size x embedding_size` matrix. However,
we can notice that the one-hot encoded vector is almost entirely 0's. If
you go through the algebra, this means that the matrix multiplication is
actually equivalent to simply returning a row from the weights matrix. That is,
for token `t`, the output is the `t`th row in the matrix. Returning a single
row from a matrix is dramatically faster than doing a matrix multiplication so
we'll do that instead. We can wrap this logic up in a new layer: [Embedding](https://github.com/bclarkson-code/Tricycle/blob/main/src/tricycle/layers.py#L365).

![embeding_layer](https://github.com/bclarkson-code/Tricycle/assets/57139598/b0157816-b797-452a-b2aa-090b3305141b)

We aren't quite done with the input block however. Transformers perform better
When they are given information about where a given token is in the context
window (e.g is a token at the start, end or somewhere in the middle?). In the
original [transformer paper](https://arxiv.org/abs/1706.03762), this was done
by with some sine waves but GPT-2 uses learned embeddings which are
conceptually simpler. (Modern language models use rotary embeddings which are
in development). When we pass a token through an embedding layer, we also pass
the index of the token through a different embedding layer and then add the two
embeddings together. This way, the embedding contains information about which
token was passed into the model, as well as where it is in the context window.

Putting these operations together, we finally get our input block:

<img width="1419" alt="input_block" src="https://github.com/bclarkson-code/Tricycle/assets/57139598/8f789407-faf6-4a7f-a3ca-593b4777604b">

#### Transformer Block

The transformer block is the core of a transformer. It is built from two main
pieces: an attention block and a multi-layer-perceptron block. Whenever we
pass some data through one of these sub-blocks, we add whatever the sub-block
outputs to the input to the block. This is called a residual layer (sometimes
also called a skip layer). I think of transformers as having a "highway" that
the embeddings pass along with each sub-block adding extra context. You can
imagine lower blocks adding information intto the embeddings that are then
read by blocks further along in the stack. Whether this mental model is helpful
remains to be seen (and I'd love to be corrected if there is something I'm
missing).

<img width="874" alt="transformer_block_high_level" src="https://github.com/bclarkson-code/Tricycle/assets/57139598/cfaf971b-662d-4ca3-b1fa-3c6786d627e0">

Gradients (derivatives) in deep learning models have a habit of rapidly
increasing in value (exploding) or decreasing to 0 (vanishing) so it is
important to frequently rescale embeddings throughout the model. You'll
notice that the embeddings are normalised before being passed through each
sub-block. In GPT-2, this is done with a [layer norm](https://github.com/bclarkson-code/Tricycle/blob/4b29bc63ec81dc22d3ff1194818e0bd2e6c095ed/src/tricycle/layers.py#L153).

#### Attention Block

If you have heard anything about transformers, it is probably that they use
attention. This is certainly the most complex part of a transformer but, at a
high level, its goal is pretty simple: let each embedding interact with the
other embeddings. This "interaction" will be in the form of a matrix, called
the attention matrix, that is `n_tokens x n_tokens x embedding_dim`. Each
entry in the matrix is a vector that represents the interaction between two
embeddings in the input.

Because this section gets a bit hairy it'll be helpful to see the goal we're
heading towards:

![attention_block](https://github.com/bclarkson-code/Tricycle/assets/57139598/17a79e58-d145-4a02-8772-c3dc98e81d2a)

The first thing we do is to pass the input embedding through a dense layer
to make each embedding 3 times longer than it used to be. Then we split the
resulting embedding into 3 separate pieces, unhelpfully called the key, query
and value vectors. Because we projected each embedding before splitting,
each of the new vectors is the same length as the original input. We won't
use the value vector until later so we'll focus on the key and query vectors
for now.

We could build our attention matrix by multiplying our key vector by our query
vector and this does work. However, in the original [transformer paper](https://arxiv.org/abs/1706.03762),
they first split each query into several smaller chunks (that they call heads)
that they compute attention matrices for individually and then recombine into
a single attention matrix at the end. They claim this improves performance
with a similar computational cost and I don't have the resources to figure out
whether this is actually true. For computational efficiency, I've avoided
explicitly splitting and recombining by doing everything inplace:

```python
# key.shape = batch_size x n_tokens x embedding_dim
# query.shape = batch_size x n_tokens x embedding_dim

head_shape = (
    self.batch_size,
    self.n_tokens,  # number of tokens
    self.n_heads,  # number of heads
    self.head_size,  # embedding per head
)

# split into multiple heads
key = key.reshape(head_shape)
query = query.reshape(head_shape)

# reorder
key = xp.einsum("BTNH->BNTH", key)
query = xp.einsum("BTNH->BNTH", query)

# attend
self.divisor = sqrt(self.head_size)
attention = xp.einsum("BNIh, BNJh -> BNIJ", query, key)
attention = attention / self.divisor
```

I'd strongly recommend having a play around with the code here to get a feel
for what these operations actually do.

Next, we need to digress slightly into how we train the model. To get our
model to generate text, we'll train it by asking it to predict the next token
in a sequence of tokens. Importantly, we do this for every token in the
sequence: token 0 in the input is used to predict token 1 in the output etc.
This means that the embeddings for earlier tokens can't be allowed to contain
information about embeddings for later tokens. Otherwise, predicting the next
token would be trivially easy for all but the final token.

Because we calculate the interaction between every token and every other token
in the attention matrix, we end up sneaking information about later tokens
into the attention for earlier tokens. To avoid this leakage, we apply a "mask"
to the attention matrix. If you work it out, you find that the leakage happens
entirely in the upper triangle of the attention matrix. We can remove this
information by manually setting each of these values to -infinity.

Finally, we normalise the matrix by softmaxing each row, multiply the
attention matrix by the value vector and reshape it to convert it back into
the original `n_tokens x embedding_dim` shape we started with. For reasons
that I'm unclear about, we pass this output through a dense layer and
optionally apply dropout if we want to regularise our model.

And that's it. My implementation of attention (without the dense layers) is
as follows:

```python
def forward(self, tensor: Tensor):
    xp = tensor.xp

    assert tensor.is_batched

    # split the input into 3 pieces
    self._input = tensor
    query = tensor[:, :, : self.embedding_dim]
    key = tensor[:, :, self.embedding_dim : self.embedding_dim * 2]
    value = tensor[:, :, self.embedding_dim * 2 :]

    # Figure out how big everything is
    self.batch_size = key.array.shape[0]
    self.head_size = self.embedding_dim // self.n_heads
    self.n_tokens = key.shape[-2]
    head_shape = (
        self.batch_size,
        self.n_tokens,  # number of tokens
        self.n_heads,  # number of heads
        self.head_size,  # embedding per head
    )
    out_shape = (self.batch_size, self.n_tokens, self.embedding_dim)

    # reshape and reorder the heads
    key = key.array
    query = query.array
    value = value.array

    key = key.reshape(head_shape)
    query = query.reshape(head_shape)
    value = value.reshape(head_shape)

    key = xp.einsum("BTNH->BNTH", key)
    query = xp.einsum("BTNH->BNTH", query)
    value = xp.einsum("BTNH->BNTH", value)

    self._key = key
    self._query = query
    self._value = value

    # attend
    self.divisor = sqrt(self.head_size)
    attention = xp.einsum("BNIh, BNJh -> BNIJ", query, key)
    attention = attention / self.divisor

    # mask
    attention = xp.where(
        self.mask[:, : self.n_tokens, : self.n_tokens], -xp.inf, attention
    )

    # softmax
    exp = xp.exp(attention - xp.max(attention, axis=-1, keepdims=True))
    denominator = xp.sum(exp, axis=-1, keepdims=True)
    attention = exp / denominator

    # TODO: come up with a better name
    # smush the heads back together
    self._before_smush = attention
    attention = xp.einsum("BNTj, BNjH -> BTNH", attention, value)
    attention = attention.reshape(out_shape)

    result = to_tensor(attention, is_batched=True)
    result.back_fns = (self.backward,)
    result.args = (self._input,)
    return result
```

Again, if you want to really understand this, I'd strongly suggest playing
around with the code to understand what each little piece does.

Splitting each vector into multiple heads make our variant of attention
"multi-head". Applying a mask to hide future tokens makes our attention
"causal" and splitting our input into 3 pieces that we then combine with each
other makes our attention "self-attention". Putting this all together, the
formal name for this variant of attention is "multi-head causal self attention".
In Tricycle, I've called it [MultiHeadSelfAttention](https://github.com/bclarkson-code/Tricycle/blob/main/src/tricycle/blocks.py#L45).

#### MLP Block

Unlike the attention block, the MLP block is much simpler. While you can think
of attention as letting different embedding vectors interact with each other,
You can think of the MLP block as adding information to each embedding
individually. First, we pass each embedding through a Dense layer that projects
it into a bigger vector. This was chosen to be 4 times longer than the
original vector in the GPT-2 paper so that's what we're using.

Next, we pass it through a non-linearity. This step is really important because
if you skip this step, mathematically, your MLP block reduces to a single (very
expensive) matrix multiplication and performance plummets. In GPT-2 we're
using [GeLU](https://github.com/bclarkson-code/Tricycle/blob/main/src/tricycle/activation.py#L24)
but I've added several other activation functions to Tricycle that you can
try out if you're interested.

Finally, we project the output back down to its original size with another
dense layer and optionally apply a dropout for regularisation.

![mlp_block](https://github.com/bclarkson-code/Tricycle/assets/57139598/e09015a5-21ff-419b-b596-d5df1a4ba728)

#### Output

Finally, once we've embedded our tokens and passed them through a stack of
transformer blocks, all that remains is to turn the embeddings back into
tokens. We can do this by passing them through a dense layer to turn each
embedding into a `1 x vocab_size` vector. We can treat each of these outputs
as a probability distribution over all tokens where larger numbers mean that
the model thinks a token is more likely to come next and smaller numbers mean
that the model thinks a token is less likely to come next.

## Building a dataset
Now we've built a model, the next stage is to build a dataset. Because we are 
building a language model, we'll start with a bunch of text data.

### Data Collection
For training GPT-2 we need a massive amount of text data. The way this is 
primarily collected is with a web-scraper that explores the public internet
and stores the content of pages that it visits. This data is often combined
with data from other sources (e.g every book and academic paper ever written).

Unfortunately, a lot of this data is bad. As I am sure the reader is well
aware, there is a lot of content on the internet that you probably don't want 
in your model. This includes NSFW content, but also includes things like 
long strings of random data or multiple copies of the same text. To fix this,
datasets are almost always passed through a variety of different filters. 

The stated goal of this project was to built a language model completely from
scratch. Maybe this means that I should have built and cleaned a dataset
myself but, in interest of time, I've decided that it isn't cheating to use
a dataset built by someone else.

As far as I can tell, the best dataset of web data for our model is
[FineWeb](https://arxiv.org/abs/2406.17557). It has a 10B token version, which is
around the right size that we'll need, and its authors claim that their 
filtering produces models that work well.

The exact blend of text data, data from other sources and filtering has a big
impact on performance which means that it is a closely guarded secret by the
big AI companies. The only way I can think of to build a dataset that produces
great models is to train several models with different blends of data and
investigate the properties of the resulting models. Because this is an
extremely expensive process, the only people that are able to do these
experiments are the big AI labs and they understandably keep the results secret.

Because nobody has let me borrow a datacenter to perform these experiments,
I'm keeping things simple with a purely web-data dataset.

### Tokenising
At time of writing, nobody has figured out how to pass text data directly to a
language model in a way that results in a working model so we instead need to
convert our text data into numbers. This is done through a process called 
tokenising.

As explained above, our GPT accepts an array of integers as an input. Each of
these integers is called a token and represents a string of data. For example,
in the tokeniser I used for GPT-2, the phrase "artificial intelligence" gets
converted into `[433, 9542, 4430]`:

![Screenshot 2024-07-11 at 12 57 52](https://github.com/bclarkson-code/Tricycle/assets/57139598/3503d3e8-7b26-44b1-8e68-286ad1bef139)

Our tokeniser needs to have a few properties:
 - It should be able to convert any string into tokens (including unicode 
characters so we can support non-english languages).
 - Common substrings should be given their own tokens (otherwise our model has
to spend a while learning which letters go after each other rather than 
learning higher level features of language)

There is an elegant algorithm that meets these requirements that tricycle uses
for tokenising called byte pair encoding.

First of all, the text is converted into an array of bytes and each individual
byte is replaced with a token. This gives us 256 unique tokens (one for each 
byte). Then, we start the training loop.

We search through our array and find the most common pair of tokens. We give
this pair a new token and then replace every ocurrence of this pair with the
new token. Then we repeat until we have the desired number of tokens.

Then, if we want to convert some new text into tokens, we can convert it into
bytes and look through our list of unique tokens to see if we can replace
any pairs with tokens. We continue this process until we can no longer replace
pairs with tokens in our list.

The full details of this algorithm can be found in the [BPETokeniser](https://github.com/bclarkson-code/Tricycle/blob/main/src/tricycle/tokeniser.py#L55). At time of writing, the tokeniser is too slow for practical use on large
datasets but development is ongoing. In the mean time, `train_smol_gpt.py` uses
OpenAI's tiktoken tokeniser.

### Getting data into the model
Now we have an array of tokens, all that remains is to feed them into the 
model. When we're training the model, we want to feed in some tokens and ask
the model to predict the next token. In practice, we select a fixed number
of tokens (the context window) and feed this into the model. Then we shift 
that context window along by one token in the dataset and use this as a label.

![Screenshot 2024-07-11 at 14 17 30](https://github.com/bclarkson-code/Tricycle/assets/57139598/773798a4-b2ee-4513-b07c-2856929ea3eb)

Now, every time we want to get some data to put into the model, we can choose
an index in our array of tokens and generate our input and label from that.
Finally, as mentioned above, performing batched operations is much faster than
single operations so we'll gather a number of input-label pairs and combine
them into two arrays, one for inputs and one for outputs. Then we can pass our
inputs into the model, calculate the loss with the labels, backpropagate and
update our weights (more on this in a bit).

## Training a Language model
At this point, we have a model and a dataset so all that remains is to
start training. The process of training a deep learning model can be split 
into 3 main peices: the forward pass, the backward pass and the weight update.

### Forward pass
The forward pass is pretty simple. We choose a batch of inputs and pass them
through the model. Because we're using Tricycle, this means that every 
operation is tracked by the automatic differentiation engine. The final output
is an array the same size as our input, but with an extra dimension. The length
of this dimension is the same as the number of unique tokens in our tokeniser.

If we passed in a `32 x 1024` array as our input, and we had 50,000 unique
tokens in our tokeniser, we would get an array of size `32 x 1024 x 50000`
as an output.

This means that each input token has a corresponding array in the output. 
Because this array is the same length as the number of tokens in our tokeniser,
we can think of each of these arrays as a score assigned to each possible
token where a larger score corresponds to the model thinking that a given
token has a larger probability of coming next.

If we had a fully trained model, we could simply choose the largest value in
each of these arrays and generate a prediction for the next token. In fact, 
we only really care about the prediction for the final token (because all of
the other predictions correspond to tokens the we already know the correct
value for because we passed them into the model. 

This means that we can select the token that has the highest score from the
final array in the output, append it to our input tokens and repeat. This is
how we generate new text with a language model! It is also why models like
ChatGPT, Claude and Gemini (if any recruiters from OpenAI, Anthropic and 
Deepmind are reading, I'm open to opportunities, so HMU) produce outputs 
word-by-word. They can only generate outputs a single token at a time.

### Backwards pass
Generating text from a language model is great, but it isn't very helpful if
the model isn't trained. For this, we need to figure out how to adjust the
model weights to produce the outputs that we want.

A good place to start is with a function called a loss function. A loss
function is just a way of measuring how correct our model outputs are. There
are a lot of different ways that we can do this but in our case, because we
are trying to select a category, a loss function that works quite well is
cross entropy.

You can find the full implementation of cross entropy in `src/tricycle/loss.py`
but at a high level, we pass in our array of scores for each token, as 
well as the correct token, and it tells us how good the predictions were in 
the form of a single number. A higher number means a worse prediction and a 
0 is a perfect prediction. 

One way to think about our loss is that it is the result of taking our input, 
passing it through a load of `Op`s (some of which also involve weights), and
eventually producing our loss. This means that, in mathematical terms, the loss
is a (very complex) function of the inputs and weights.

Because every `Op` in Tricycle is differentiable, we're able to differentiate
our loss w.r.t each of the weights in the network. This is a really powerful
idea because there is a theorem in vector calculus that says that the 



## What's Next?

 - Documentation
    [ ] Explain how to train a language model
    [ ] Explain the tokeniser

 - Code
    [ ] Rotary Embeddings
    [ ] Test RMS Norm
    [ ] Multi-GPU support
    [ ] Optimise and use the tokeniser

 - Experiments
    [ ] Try a language dataset rather than pure code
    [ ] Build a LLama style model
    [ ] Build a bigger langauge model (GPT-2 sized?)

<!-- ### Training a Language model -->
<!---->
<!-- Now we've built our langauge model, we need to actually train it.  -->

## Contact

Want to work together? You can reach me at: [bclarkson-code@proton.me](mailto:bclarkson-code@proton.me)
