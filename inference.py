import pickle
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

from tricycle.configs import SmolGPTConfig
from tricycle.functions import softmax
from tricycle.layers import Dropout
from tricycle.models import GPT
from tricycle.tensor import to_tensor
from tricycle_datasets.shakespeare import Shakespeare, ShakespeareChar

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


def load_model(version):
    print(
        f"LOADING MODEL: model_b30dc841-05f2-4e48-b0bc-49c2401f19e3_{version}"
    )
    with open(
        f"models/model_b30dc841-05f2-4e48-b0bc-49c2401f19e3_{version}.pkl",
        "rb",
    ) as f:
        return pickle.load(f)


def deactivate_dropout(model):
    stack = [model]

    while stack:
        node = stack.pop()
        if isinstance(node, Dropout):
            node.probability = 0

        if not node.layers:
            continue

        stack.extend(iter(node.layers))


def generate(text, model, tokeniser, sample=True):
    tokens = tokeniser.encode(text)
    while True:
        tokens = tokens[-config.context_window :]
        n_tokens = len(tokens)
        tokens = pad(tokens)

        encoded = to_tensor(
            [tokens], dtype=int, requires_grad=False
        ).to_vector()

        pred, _ = model(encoded)
        pred = softmax(pred)
        probabilities = pred.xp.asnumpy(pred._data[0][n_tokens - 1])

        # sample according to probabilities
        if sample:
            next_token = np.random.choice(
                list(range(config.vocab_size)), p=probabilities
            )
        else:
            next_token = np.argmax(probabilities)
        tokens.append(next_token)
        yield next_token


def get_sample(sample_text, model, tokeniser, n_samples=50):
    sampled = []
    for i, next_token in enumerate(generate(sample_text, model, tokeniser)):
        if i > n_samples:
            break
        sampled.append(next_token)
    model.zero_grad()
    return tokeniser.decode(sampled)


if __name__ == "__main__":
    np.random.seed(0)

    config = SmolGPTConfig()
    shakespeare = ShakespeareChar()
    shakespeare.vocab_size = 65

    tokeniser = shakespeare
    try:
        version = int(sys.argv[1])
    except:
        version = 2750
    model = load_model(version)
    model.to_gpu(1)

    deactivate_dropout(model)

    sample_text = """'er my head
As is a winged messenger of heaven
Unto the white-upturned wondering eyes
Of mortals that fall back to gaze on him
When he bestrides the lazy-pacing clouds
And sails upon the bosom of the air.

JULIET:
O Romeo, Romeo! wherefore art thou Romeo?
"""
    print(
        f"------------PROMPT-------------\n{sample_text}\n--------------RESPONSE-----------",
        flush=True,
    )
    sys.stdout.flush()
    for token in generate(sample_text, model, tokeniser, sample=True):
        token = int(token)
        token = tokeniser.decode([token])[0]
        token = chr(token)
        print(token, end="", flush=True)
