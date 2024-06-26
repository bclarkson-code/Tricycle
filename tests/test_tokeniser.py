import numpy as np
import pytest

from tricycle.tokeniser import BPETokeniser, count_pairs, replace_pair

slow_test = pytest.mark.skipif(
    "not config.getoption('--run-slow')",
    reason="Only run when --run-slow is given",
)


def test_count_pairs():
    data = np.array([0, 0, 1, 0, 1], dtype=np.int32)

    got = count_pairs(data=data, token_id=1)
    want = np.array([1, 2, 1, 0])

    assert np.allclose(got, want)


def test_replace_pair():
    to_replace = (1, 2)
    data = np.array([1, 1, 2, 1, 2, 1])

    got = replace_pair(data, to_replace, 3)
    want = np.array([1, 3, 3, 1])

    assert np.allclose(got, want)


def test_replace_pair_when_final_tokens_are_pair():
    tokeniser = BPETokeniser(256)

    to_replace = (1, 2)
    data = np.array([1, 1, 2, 1, 2])

    got = tokeniser.replace_pair(data, to_replace, 3)
    want = np.array([1, 3, 3])

    assert np.allclose(got, want)


def test_can_train_simple_text():
    tokeniser = BPETokeniser(256 + 3)
    sample_text = "aababa"

    with pytest.warns(UserWarning):
        tokeniser.train(sample_text)

    assert tokeniser.vocab_size == 256 + 3

    assert (ord("a"), ord("b")) in tokeniser.merges
    assert (ord("a"), ord("b")) in tokeniser.pairs

    assert len(tokeniser.merges) == len(tokeniser.pairs) == 257


def test_can_tokenise_simple_text():
    tokeniser = BPETokeniser(257)
    tokeniser.merges[(ord("a"), ord("b"))] = 256

    sample_text = "aababa"
    got = tokeniser.encode(sample_text)
    want = np.array([ord("a"), 256, 256, ord("a")])

    assert np.allclose(got, want)


def test_can_tokenise_paragraph():
    tokeniser = BPETokeniser(300)

    sample_text = """(Barry is picking out a shirt)
Yellow, black. Yellow, black.
Yellow, black. Yellow, black.
 :
Ooh, black and yellow!
Let's shake it up a little.
"""
    tokeniser.train(sample_text)
    got = tokeniser.encode(sample_text)
    want = np.array(
        [
            40,
            66,
            97,
            114,
            114,
            121,
            271,
            115,
            32,
            112,
            105,
            256,
            105,
            110,
            103,
            32,
            111,
            117,
            116,
            269,
            275,
            105,
            114,
            116,
            41,
            274,
            274,
            10,
            32,
            58,
            10,
            79,
            111,
            104,
            263,
            269,
            110,
            100,
            32,
            121,
            265,
            33,
            10,
            76,
            101,
            116,
            39,
            115,
            275,
            97,
            107,
            101,
            271,
            116,
            32,
            117,
            112,
            269,
            32,
            108,
            105,
            116,
            116,
            108,
            101,
            46,
            10,
        ]
    )
    assert np.allclose(got, want)


def test_can_decode_tokens():
    tokeniser = BPETokeniser(257)
    tokeniser.vocab.append(b"ab")

    sample_tokens = np.array([ord("a"), 256, 256, ord("a")])
    got = tokeniser.decode(sample_tokens)
    want = "aababa"

    assert got == want


def test_can_tokenise_longer_text():
    tokeniser = BPETokeniser(1000)

    with open("datasets/bee_movie.txt", "r") as f:
        sample_text = f.read()

    tokeniser.train(sample_text)

    assert len(tokeniser.merges) == len(tokeniser.pairs) == 1000

    got = tokeniser.encode(sample_text)

    assert np.allclose(got[:10], [78, 279, 82, 65, 84, 829, 684, 66, 337, 386])
    assert np.allclose(
        got[-10:], [644, 617, 339, 454, 266, 115, 600, 437, 468, 262]
    )
