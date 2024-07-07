import pickle
import sys
from pathlib import Path

import numpy as np
import tiktoken
from tqdm import tqdm

from tricycle.configs import SmolGPTConfig
from tricycle.functions import Softmax
from tricycle.layers import Dropout, Layer
from tricycle.models import GPT
from tricycle.tensor import Tensor
from tricycle.tokeniser import BPETokeniser
from tricycle_datasets.fineweb import FineWeb

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


def generate(
    model: GPT,
    tokens: np.ndarray | None = None,
    sample=True,
    temperature=0.8,
):
    """
    Given a prompt, yield next token predictions for a model
    """
    if isinstance(tokens, np.ndarray):
        tokens = tokens.tolist()

    while True:
        tokens = tokens[-config.context_window :]
        assert len(tokens) == config.context_window

        encoded = Tensor(
            [tokens], dtype=np.uint16, requires_grad=False, is_batched=True
        )

        pred = model(encoded)
        pred = Softmax()(pred / temperature)

        if pred.on_gpu:
            probabilities = pred.xp.asnumpy(
                pred.array[0][config.context_window - 1]
            )
        else:
            probabilities = pred.array[0][config.context_window - 1]

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
    tokeniser: BPETokeniser | tiktoken.core.Encoding,
    sample_tokens: np.ndarray | None = None,
) -> str:
    """
    Given a prompt, generate some new tokens and return them as a string
    """
    sampled = []
    for i, next_token in tqdm(
        enumerate(
            generate(
                tokens=sample_tokens,
                model=model,
            )
        ),
        desc="Sampling",
        total=config.n_tokens_to_generate,
        position=1,
        leave=False,
    ):
        if i > config.n_tokens_to_generate:
            break
        sampled.append(next_token)

    decoded = tokeniser.decode(sampled)
    sample_text = tokeniser.decode(sample_tokens)
    decoded = f"PROMPT:\n{sample_text}\nGENERATED:\n{decoded}"
    return decoded


if __name__ == "__main__":
    np.random.seed(0)

    # config = ShakespeareConfig()
    # dataset = Shakespeare(config.vocab_size)
    config = SmolGPTConfig()
    dataset = FineWeb(config.vocab_size, split="valid")

    import cupy

    model = load_model(sys.argv[1])
    model.to_gpu(0)

    # model.from_gpu()
    model.zero_grad()
    deactivate_dropout(model)

    # with open('models/smol_gpt_fineweb.pkl', 'wb') as f:
    #     pickle.dump(model, f)

    START = 5_000
    sample_tokens = dataset.tokens[START : START + 1024]
    line = []
    prev = ""
    for token in generate(tokens=sample_tokens, model=model, sample=True):
        if token == dataset.tokeniser.eot_token:
            break
        token = int(token)
        line = line + [token]
        new = dataset.decode(line)
        if ord(new[-1]) != 65533:
            diff = new[len(prev) :]
            prev = new

            print(diff, end="", flush=True)
