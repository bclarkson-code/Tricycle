# Tricycle
Tricycle is a fast, minimal, fully functional deep learning library written from scratch using only python and numpy.
The file 'train_smol_gpy.py' trains a 49M, GPT-2 style language model to produce python code in ~2 days on my RTX 3090 (24Gb).

The entire library, from the automatic differentiation engine to a GPT, is written in ~4500 lines of python + numpy code.
Using [CuPY](https://cupy.dev/), all Tricycle code can run on a GPU and is only about ~TODO: insert comparision to pytorch here.~ % [slower than pytorch](#comparison-with-pytorch).

## Installation
Tricycle uses [conda](https://docs.conda.io/en/latest/) to manage dependencies. While we do support CPU-only computation, optimisation efforts have been focussed on GPU computation so it is pretty slow. If you do have a CUDA capable GPU I would strongly recommend installing the gpu version of Tricycle.

If you have a CUDA capable GPU, you can install Tricycle as follows.
```bash
conda env create -f environment.yml -n tricycle
conda activate tricycle
```

<details>
    <summary>Other installations</summary>
If you want to install test-dependencies you can do the following.

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

If you want to install test-dependencies you can do the following.
```bash
conda env create -f environment.cpu.test.yml -n tricycle
conda activate tricycle
```
</details>



## Features


## Training a GPT on shakespeare
The following toy script will train a small GPT to generate convincing shakespeare.
On my RTX 3090, this takes ~30 mins. For a more realistic training script with metric tracking, gradient accumulation etc, take a look at `train_smol_gpt.py`

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
Once trained, you can generate new shakespeare as follows:

```
python inference.py model.pkl
```


## Contact
Want to learn more / have a chat / work together?
You can send an email to: [bclarkson-code@proton.me](mailto:bclarkson-code@proton.me)
