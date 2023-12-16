from llm_from_scratch.ops import tensor


def test_can_mean():
    assert tensor([1, 2, 3]).mean() == 2
