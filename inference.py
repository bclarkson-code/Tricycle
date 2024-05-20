import pickle
import sys
from pathlib import Path

import numpy as np

from tricycle.configs import SmolGPTConfig
from tricycle.functions import Softmax
from tricycle.layers import Dropout, Layer
from tricycle.tensor import to_tensor
from tricycle_datasets.shakespeare import Shakespeare, ShakespeareChar

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


def generate(text, model, tokeniser, sample=True, temperature=0.8):
    """
    Given a prompt, yield next token predictions for a model
    """
    tokens = tokeniser.encode(text)
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
    shakespeare = Shakespeare(config.vocab_size)

    tokeniser = shakespeare
    model = load_model(sys.argv[1])
    model.to_gpu(0)

    deactivate_dropout(model)

    sample_text = """ROMEO:
He jests at scars that never felt a wound.
But, soft! what light through yonder window breaks?
It is the east, and Juliet is the sun.
Arise, fair sun, and kill the envious moon,
Who is already sick and pale with grief,
That thou her maid art far more fair than she:
Be not her maid, since she is envious;
Her vestal livery is but sick and green
And none but fools do wear it; cast it off.
It is my lady, O, it is my love!
O, that she knew she were!
She speaks yet she says nothing: what of that?
Her eye discourses; I will answer it.
I am too bold, 'tis not to me she speaks:
Two of the fairest stars in all the heaven,
Having some business, do entreat her eyes
To twinkle in their spheres till they return.
What if her eyes were there, they in her head?
The brightness of her cheek would shame those stars,
As daylight doth a lamp; her eyes in heaven
Would through the airy region stream so bright
That birds would sing and think it were not night.
See, how she leans her cheek upon her hand!
O, that I were a glove upon that hand,
That I might touch that cheek!

JULIET:
Ay me!

ROMEO:
She speaks:
O, speak again, bright angel! for thou art
As glorious to this night, being o'er my head
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
        token = tokeniser.decode([token])
        print(token, end="", flush=True)
