import pickle
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

from tricycle.configs import SmolGPTConfig
from tricycle.functions import Softmax
from tricycle.layers import Dropout, Layer
from tricycle.models import GPT
from tricycle.tensor import to_tensor
from tricycle_datasets.codeparrot import CodeParrot
from tricycle_datasets.shakespeare import Shakespeare

config = SmolGPTConfig()


def load_model(path: str | Path) -> Layer:
    print(f"LOADING MODEL: {path}")
    with open(
        path,
        "rb",
    ) as f:
        return pickle.load(f)


def deactivate_dropout(model: Layer) -> Layer:
    """
    Traverse through the model and deactivate any dropout layers
    """
    stack = [model]

    while stack:
        node = stack.pop()
        if isinstance(node, Dropout):
            node.probability = 0

        if not node.layers:
            continue

        stack.extend(iter(node.layers))
    return model


# TODO: allow tokensiers that arent shakespeare
def generate(
    model: GPT,
    tokeniser: Shakespeare,
    text: str | None = None,
    tokens: np.ndarray | None = None,
    sample=True,
    temperature=0.8,
):
    """
    Given a prompt, yield next token predictions for a model
    """
    if text is not None:
        tokens = tokeniser.encode(text)
    elif tokens is None:
        raise ValueError("At least one of text, tokens must not be None")

    while True:
        tokens = tokens[-config.context_window :]
        assert len(tokens) == config.context_window

        encoded = to_tensor(
            [tokens], dtype=int, requires_grad=False
        ).to_vector()

        pred = model(encoded)
        pred = Softmax()(pred / temperature)

        if pred.on_gpu:
            probabilities = pred.xp.asnumpy(
                pred._data[0][config.context_window - 1]
            )
        else:
            probabilities = pred._data[0][config.context_window - 1]

        # sample according to probabilities
        if sample:
            next_token = np.random.choice(
                list(range(config.vocab_size)), p=probabilities
            )
        else:
            next_token = np.argmax(probabilities)
        tokens.append(next_token)
        yield next_token


def get_sample(
    model: GPT,
    dataset: Shakespeare | CodeParrot,
    sample_text: str | None = None,
    n_samples: int = 50,
) -> str:
    """
    Given a prompt, generate some new tokens and return them as a string
    """
    if sample_text is None:
        # we need a full context window before we start generating so this
        # text is more than we'll need
        sample_text = dataset.tokens[:2048]
    sampled = []
    for i, next_token in tqdm(
        enumerate(generate(text=sample_text, model=model, tokeniser=dataset)),
        desc="evaluating",
        total=n_samples,
        position=1,
        leave=False,
    ):
        if i > n_samples:
            break
        sampled.append(next_token)
    decoded = dataset.decode(sampled)
    if not isinstance(decoded, str):
        decoded = "".join([chr(i) for i in decoded])
    return decoded


if __name__ == "__main__":
    np.random.seed(0)

    config = SmolGPTConfig()
    dataset = Shakespeare(config.vocab_size)

    model = load_model(sys.argv[1])
    model.to_gpu(0)

    deactivate_dropout(model)

    sample_text = dataset.raw_data_path.read_text()[:2048]
    for token in generate(
        text=sample_text, model=model, tokeniser=dataset, sample=True
    ):
        token = int(token)
        token = dataset.decode([token])
        print(token, end="", flush=True)
