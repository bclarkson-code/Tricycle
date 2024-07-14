# Tricycle

<p align="center">
    <img width="223" alt="tricycle_logo" src="https://github.com/bclarkson-code/Tricycle/assets/57139598/62405944-b27b-49bc-93c3-17ba93fc8ad7">
</p>

Tricycle is a fast, minimal, fully functional deep learning library written from scratch using only python and numpy.

While I've tried to make it easy to follow, Tricycle is not just another toy
neural network: the file `train_smol_gpt.py` trains GPT-2 (124M) on 2.5B
tokens in just under 3 days on my GPU (RTX 3090).


The entire library, from the automatic differentiation engine to a GPT,
should be understandable to anyone with a bit of python experience and I
strongly encourage you to have a play around with the codebase to understand
how everything works.

All Tricycle code can run on either a CUDA-capable GPU or a CPU (although
optimisation efforts have been focussed on the GPU so CPU computation is
really slow).

## Table of Contents
- [Tricycle](#tricycle)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Training a GPT on Shakespeare](#training-a-gpt-on-shakespeare)
  - [Training GPT-2 (124M)](#training-gpt-2-124m)
  - [How it works](#how-it-works)
  - [What's Next?](#whats-next)
  - [Contact](#contact)

## Installation

Tricycle uses [conda](https://docs.conda.io/en/latest/) to manage dependencies.
While we do support CPU-only computation, optimisation efforts have been
focussed on GPU computation so it is pretty slow. If you do have a CUDA
capable GPU I would strongly recommend installing the gpu version of Tricycle.

If you have a CUDA capable GPU, you can install Tricycle as follows.

```bash
conda env create -f requirements/environment.yml -n tricycle
conda activate tricycle
```

Otherwise, you can install the cpu-only version like this:

```bash
conda env create -f requirements/environment.cpu.yml -n tricycle
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

## Training GPT-2 (124M)
To train GPT-2, you'll first want to start an MLFlow server in a separate
terminal for tracking metrics during training. You can do this as follows:
```bash
mlflow server --host 0.0.0.0
```

Then should be able to just run
```bash
python train_smol_gpt.py
```
If you have a CUDA capable GPU, training should start immediately.

The parameters for the model can be found at `src/tricycle/configs.py:SmolGPTConfig`.
If you aren't using an RTX 3090, you'll probably want to play around with the
parameters a bit to optimise things for your setup (e.g increasing batch size
if you have a GPU with more VRAM).

## How it works
Tricycle started as a thin wrapper around a numpy array and has developed into
a modern deep learning framework. If you'd like to know how, check out the
[Tricycle Wiki](https://github.com/bclarkson-code/Tricycle/wiki/How-it-works)

## What's Next?

 - Documentation
    [ ] Explain how to train a language model
    [X] Explain the tokeniser

 - Code
    [ ] Rotary Embeddings
    [ ] Test RMS Norm
    [ ] Multi-GPU support
    [ ] Optimise and use the tokeniser

 - Experiments
    [X] Try a language dataset rather than pure code
    [ ] Build a LLama style model
    [X] Build a bigger langauge model (GPT-2 sized?)

<!-- ### Training a Language model -->
<!---->
<!-- Now we've built our langauge model, we need to actually train it.  -->

## Contact

Want to work together? You can reach me at: [bclarkson-code@proton.me](mailto:bclarkson-code@proton.me)
