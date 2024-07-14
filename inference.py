import argparse
import pickle
from copy import copy
from pathlib import Path

import numpy as np
import tiktoken
from tqdm import tqdm

from tricycle.configs import DebugConfig, ShakespeareConfig, SmolGPTConfig
from tricycle.functions import Softmax
from tricycle.layers import Dropout, Layer
from tricycle.models import GPT
from tricycle.tensor import Tensor
from tricycle.tokeniser import BPETokeniser
from tricycle_datasets.fineweb import FineWeb
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


def generate(
    model: GPT,
    tokens: np.ndarray | None = None,
    sample=True,
    temperature=0.8,
    pad_token=-1,
):
    """
    Given a prompt, yield next token predictions for a model
    """
    if isinstance(tokens, np.ndarray):
        tokens = tokens.tolist()

    while True:
        tokens = tokens[-config.context_window :]
        n_tokens = len(tokens)
        if n_tokens < config.context_window:
            pad_tokens = [pad_token] * (config.context_window - n_tokens)
            tokens += pad_tokens

        encoded = Tensor(
            tokens, dtype=np.uint32, requires_grad=False, is_batched=False
        )

        pred = model(encoded)
        pred = Softmax()(pred / temperature)

        next_token_idx = n_tokens - 1

        if pred.on_gpu:
            probabilities = pred.xp.asnumpy(pred.array[0][next_token_idx])
        else:
            probabilities = pred.array[0][next_token_idx]

        # sample according to probabilities
        if sample:
            next_token = np.random.choice(
                list(range(config.vocab_size)), p=probabilities
            )
        else:
            next_token = np.argmax(probabilities)

        # remove padding + add new token
        tokens = tokens[:n_tokens]
        tokens.append(next_token)

        # convert from numpy int to python int
        yield int(next_token)


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
    parser = argparse.ArgumentParser(
        prog="inference.py", description="Generate predictions from a GPT"
    )

    parser.add_argument("model_path")
    parser.add_argument("prompt", help="Text that will be passed to the model")
    parser.add_argument(
        "-c",
        "--model_config",
        choices=["debug", "smol_gpt", "shakespeare"],
        default="shakespeare",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        choices=["shakespeare", "fineweb"],
        default="shakespeare",
    )
    parser.add_argument("--use-gpu", action='store_true')

    args = parser.parse_args()
    print(args)

    match args.model_config:
        case "shakespeare":
            config = ShakespeareConfig()
        case "smol_gpt":
            config = SmolGPTConfig()
        case "debug":
            config = DebugConfig()
        case _:
            raise ValueError(f"Unknown dataset: {args.config}")

    match args.dataset:
        case "shakespeare":
            dataset = Shakespeare(config.vocab_size)
        case "fineweb":
            dataset = FineWeb(config.vocab_size, split="valid")
        case _:
            raise ValueError(f"Unknown dataset: {args.dataset}")

    np.random.seed(0)

    model_path = Path(args.model_path)
    if model_path.exists():
        model = load_model(model_path)
    else:
        raise FileNotFoundError(
            f"Could not find model file: {model_path.absolute()}"
        )

    if args.use_gpu:
        model.to_gpu(0)

    model.zero_grad()
    deactivate_dropout(model)

    sample_tokens = dataset.tokeniser.encode(args.prompt)
    if isinstance(sample_tokens, np.ndarray):
        sample_tokens = sample_tokens.tolist()
    generated = copy(sample_tokens)
    prev = args.prompt
    for token in generate(
        tokens=sample_tokens, model=model, sample=True, pad_token=0
    ):
        if args.dataset == "fineweb" and token == dataset.tokeniser.eot_token:
            break
        generated += [token]
        try:
            if isinstance(dataset.tokeniser, BPETokeniser):
                decoded = dataset.tokeniser.decode(np.array(generated))
            else:
                decoded = dataset.tokeniser.decode(generated, errors="strict")
        except UnicodeDecodeError:
            new = decoded[len(prev) :]
            prev = decoded
            continue

        new = decoded[len(prev) :]
        print(new, end="", flush=True)
        prev = decoded
