import pickle
from pathlib import Path

import numpy as np

from tricycle.configs import SmolGPTConfig
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


def load_model():
    with open("shakespeare_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model


def one_hot_encode(tokens):
    """
    One hot encode some tokens into one-hot vectors
    """
    one_hot = np.zeros((len(tokens), config.vocab_size))

    for i, token in enumerate(tokens):
        one_hot[i, token] = 1
    return one_hot


def detokenise(tokens, tokeniser):
    undo_tokenising = {v: k for k, v in tokeniser.merges.items()}
    all_bytes = False

    while not all_bytes:
        detokenised = []
        all_bytes = True
        for t in tokens:
            if t < 256:
                detokenised.append(t)
            else:
                all_bytes = False
                detokenised.extend(list(undo_tokenising[t]))
        tokens = detokenised
    return bytes(tokens)


if __name__ == "__main__":
    tokeniser = load_tokeniser()

    model = load_model()
    model.input_dropout.probability = 0
    for block in model.blocks:
        block.attention_block.attention_dropout.probabillity = 0
        block.attention_block.residual_dropout.probabillity = 0
        block.mlp_block.dropout.probability = 0

    sample_text = "To be or not to be, that is the question"
    tokens = tokeniser.tokenise(sample_text)
    print(tokens)

    start = max(len(tokens) - config.context_window, 0)
    end = len(tokens)

    if len(tokens) < config.context_window:
        n_padding_tokens = config.context_window - len(tokens)
        tokens += [0] * n_padding_tokens

    encoded = np.expand_dims(one_hot_encode(tokens), 0)
    encoded = to_tensor(encoded).to_vector()

    pred = model(encoded).numpy()
    print(pred.shape)
    print(np.argmax(pred[0][0]))
    tokens = np.argmax(pred, axis=-1)[0]
    print(tokens.shape)
    print(tokens)
    text = detokenise(tokens, tokeniser)
    print(text)
