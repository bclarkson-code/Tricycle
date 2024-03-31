import pytest

from tricycle.tokeniser import BPETokeniser

slow_test = pytest.mark.skipif(
    "not config.getoption('--run-slow')",
    reason="Only run when --run-slow is given",
)


def test_count_pairs():
    tokeniser = BPETokeniser(256)
    data = [0, 0, 1, 0, 1]

    got = tokeniser.count_pairs(data)
    want = {
        (0, 0): 1,
        (0, 1): 2,
        (1, 0): 1,
    }
    assert got == want


def test_replace_pair():
    tokeniser = BPETokeniser(256)

    to_replace = (1, 2)
    data = [1, 1, 2, 1, 2, 1]

    got = tokeniser.replace_pair(data, to_replace, 3)
    want = [1, 3, 3, 1]

    assert got == want


def test_replace_pair_when_final_tokens_are_pair():
    tokeniser = BPETokeniser(256)

    to_replace = (1, 2)
    data = [1, 1, 2, 1, 2]

    got = tokeniser.replace_pair(data, to_replace, 3)
    want = [1, 3, 3]

    assert got == want


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
    got = tokeniser.tokenise(sample_text)
    want = [ord("a"), 256, 256, ord("a")]

    assert got == want


@slow_test
def test_can_tokenise_longer_text():
    tokeniser = BPETokeniser(1000)

    with open("datasets/bee_movie.txt", "r") as f:
        sample_text = f.read()

    tokeniser.train(sample_text)

    assert len(tokeniser.merges) == len(tokeniser.pairs) == 1000

    got = tokeniser.tokenise(sample_text)

    assert got[:10] == [78, 279, 82, 65, 693, 82, 675, 66, 337, 383]
    assert got[-10:] == [640, 612, 339, 455, 266, 115, 597, 434, 464, 262]