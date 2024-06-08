import numpy as np

from tricycle.dataset import CausalLMDataset


def test_can_build_causal_lm_dataset():
    tokens = np.arange(100)
    dataset = CausalLMDataset(
        tokens=tokens, vocab_size=100, batch_size=10, context_window=10
    )

    inputs, outputs = dataset[0]
    assert isinstance(inputs, np.ndarray)
    assert isinstance(outputs, np.ndarray)

    assert len(inputs) == 10
    expected_tokens = tokens[:11]

    assert np.allclose(inputs, expected_tokens[:-1])
    assert np.allclose(outputs, expected_tokens[1:])

    dataset.batch()

    inputs, outputs = dataset[0]
    assert inputs.shape == (10, 10)
    assert outputs.shape == (10, 10)
