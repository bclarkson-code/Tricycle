# Tricycle
Tricycle is a fast, minimal, fully functional deep learning library written from scratch using only python and numpy.

The file `train_smol_gpy.py` trains a 49M parameter, GPT-2 style language model to produce python code in ~2 days on a single RTX 3090.

The entire library, from the automatic differentiation engine to a GPT, is written in ~4500 lines of python + numpy code.

Using [CuPY](https://cupy.dev/), all Tricycle code can run on a GPU and is only about ~TODO: insert comparision to pytorch here.~ % [slower than pytorch](#comparison-with-pytorch). Tricycle is still under active development so this is subject to change.

## Installation
Tricycle uses [conda](https://docs.conda.io/en/latest/) to manage dependencies. While we do support CPU-only computation, optimisation efforts have been focussed on GPU computation so it is pretty slow. If you do have a CUDA capable GPU I would strongly recommend installing the gpu version of Tricycle.

If you have a CUDA capable GPU, you can install Tricycle as follows.
```bash
conda env create -f environment.yml -n tricycle
conda activate tricycle
```

<details>
    <summary>CPU and test installation</summary>
If you want to install test dependencies you can do the following.

```bash
conda env create -f environment.test.yml -n tricycle
conda activate tricycle
```

### CPU Installation
If you want to install Tricycle for CPU, you can do the following.
```bash
conda env create -f environment.cpu.yml -n tricycle
conda activate tricycle
```

If you want to install test dependencies on CPU you can do the following.
```bash
conda env create -f environment.cpu.test.yml -n tricycle
conda activate tricycle
```
</details>


## Training a GPT on shakespeare
The following toy script will train a small GPT to generate convincing shakespeare.
On my RTX 3090, this takes ~30 mins. For a more realistic training script with metric tracking, gradient accumulation, a validation dataset etc, take a look at `train_smol_gpt.py`

```python
import pickle

from tqdm import tqdm

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

model.to_gpu()
loading_bar = tqdm(range(config.steps))
for step in loading_bar:
    optimiser.step()
    inputs, outputs = next(dataset)
    inputs = inputs.to_gpu()
    outputs = outputs.to_gpu()

    logits = model(inputs)
    loss = loss_fn(outputs, logits)
    loss.backward()

    loading_bar.set_description(f"loss: {loss:.3f}")
    model.update(optimiser)

# save results
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
```
Once trained, you can generate infinite shakespeare plays as follows:

```bash
python inference.py model.pkl
```

## How it works
Tricycle code centers around objects called `Tensors`. A `Tensor` is a wrapper around a numpy array that adds some extra features:

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
Unlike vanilla numpy, every operation in Tricycle is attached to a derivative.
When you do some operations on your `Tensor`, Tricycle keeps track of what you did and allows you to differentiate the output.

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
input = to_tensor(np.ones(shape), is_vector=True)

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

The entirety of the algorithm can be found in [`tensor.py`](https://github.com/bclarkson-code/Tricycle/blob/update-readme/src/tricycle/tensor.py#L145).
It ends up being a topological sort to figure out which order to traverse the tree and then a simple traversal, applying the `back_fns` along the way.

If you want a more detailed explanation, I've talked about it on [my blog](https://bclarkson-code.com/posts/llm-from-scratch-scalar-autograd/post.html).

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

### Building a Neural network

Now we have the automatic differentiation engine built, we can start building the components of a neural network.
We can start with a [Dense Layer](https://github.com/bclarkson-code/Tricycle/blob/update-readme/src/tricycle/layers.py#L34).

## Contact
Want to work together? You can reach me at: [bclarkson-code@proton.me](mailto:bclarkson-code@proton.me)
