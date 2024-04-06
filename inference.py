import pickle
import sys
from pathlib import Path
from typing import Sequence

import cupy as cp
import numpy as np

from tricycle.configs import SmolGPTConfig
from tricycle.functions import softmax
from tricycle.layers import Dropout
from tricycle.tensor import to_tensor
from tricycle_datasets.shakespeare import Shakespeare

config = SmolGPTConfig()


def load_tokeniser():
    tokeniser_path = Path("tokeniser.pkl")
    if not tokeniser_path.exists():
        dataset = Shakespeare(vocab_size=1024, token_path=Path("invalid_path"))
        tokeniser = dataset.tokeniser
        with open(tokeniser_path, "wb") as f:
            pickle.dump(tokeniser, f)
    else:
        with open(tokeniser_path, "rb") as f:
            tokeniser = pickle.load(f)
    return tokeniser


def one_hot_encode(
    tokens: Sequence[int],
    vocab_size=config.vocab_size,
    pad_token_id=config.pad_token_id,
):
    """
    One hot encode some tokens into one-hot vectors
    """
    one_hot = np.zeros((len(tokens), vocab_size))

    for i, token in enumerate(tokens):
        if token == pad_token_id:
            continue
        one_hot[i, token] = 1
    return one_hot


def pad(
    tokens: list[int],
    context_window=config.context_window,
    pad_token_id=config.pad_token_id,
):
    n_padding_tokens = max(0, context_window - len(tokens))
    return tokens + [pad_token_id] * n_padding_tokens


def load_model():
    with open("models/shakespeare_model_mmmmmmmmmmm.pkl", "rb") as f:
        model = pickle.load(f)
    return model


def deactivate_dropout(model):
    stack = [model]

    while stack:
        node = stack.pop()
        if isinstance(node, Dropout):
            node.probability = 0

        if not node.layers:
            continue

        stack.extend(iter(node.layers))


def generate(text, model, sample=True):
    tokens = tokeniser.encode(text)
    while True:
        n_tokens = len(tokens)
        tokens = pad(tokens)

        encoded = one_hot_encode(tokens)
        encoded = np.expand_dims(one_hot_encode(tokens), 0)
        encoded = to_tensor(encoded).to_vector()

        pred = model(encoded)
        pred = softmax(pred)
        probabilities = cp.asnumpy(pred._data[0][n_tokens])

        # sample according to probabilities
        if sample:
            next_token = np.random.choice(
                list(range(config.vocab_size)), p=probabilities
            )
        else:
            next_token = np.argmax(probabilities)
        tokens = tokens[:n_tokens]
        yield next_token


if __name__ == "__main__":
    np.random.seed(0)
    tokeniser = load_tokeniser()
    model = load_model()

    deactivate_dropout(model)

    sample_text = "Here is some example text"
    print(sample_text, end="", flush=True)
    sys.stdout.flush()
    for token in generate(sample_text, model, sample=False):
        print(tokeniser.decode(token), end="", flush=True)
