import numpy as np

from tricycle.dataset import CausalLMDataset


def test_can_build_causal_lm_dataset():
    tokens = list(range(100))
    dataset = CausalLMDataset(
        tokens, vocab_size=100, batch_size=10, context_window=10
    )

    inputs, output = dataset._get_single(0)

    assert len(inputs) == 10
    expected_tokens = tokens[:11]
    expected_vectors = np.zeros((11, 100))
    expected_vectors[np.arange(11), expected_tokens] = 1

    assert np.allclose(inputs, expected_vectors[:10])
    assert np.allclose(output, expected_vectors[10])

    dataset.batch()

    inputs, outputs = dataset._get_batch(0)
    assert inputs.shape == (10, 10, 100)
    assert outputs.shape == (10, 100)
