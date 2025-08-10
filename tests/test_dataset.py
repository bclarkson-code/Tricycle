from unittest.mock import MagicMock, patch

import numpy as np

from tricycle.dataset import CausalLMDataset, MmappedCausalLMDataset


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


def test_dataset_initialization():
    tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    dataset = MmappedCausalLMDataset(
        tokens=tokens,
        vocab_size=100,
        batch_size=2,
        context_window=3,
    )

    assert dataset.n_tokens == 10
    assert dataset.vocab_size == 100
    assert dataset.batch_size == 2
    assert dataset.context_window == 3
    assert dataset.is_batch == False
    assert dataset.as_tensor == False
    assert dataset._idx == 0
    assert dataset.filename.exists()


def test_dataset_length():
    tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    dataset = MmappedCausalLMDataset(
        tokens=tokens,
        vocab_size=100,
        batch_size=2,
        context_window=3,
    )

    expected_length = 10 - 3 - 1
    assert len(dataset) == expected_length


def test_single_item_retrieval():
    tokens = [10, 20, 30, 40, 50, 60, 70, 80]
    dataset = MmappedCausalLMDataset(
        tokens=tokens,
        vocab_size=100,
        batch_size=2,
        context_window=3,
    )

    inputs, outputs = dataset[0]

    assert isinstance(inputs, np.ndarray)
    assert isinstance(outputs, np.ndarray)
    assert len(inputs) == 3
    assert len(outputs) == 3
    np.testing.assert_array_equal(inputs, [10, 20, 30])
    np.testing.assert_array_equal(outputs, [20, 30, 40])


def test_single_item_out_of_bounds():
    tokens = [1, 2, 3, 4, 5]
    dataset = MmappedCausalLMDataset(
        tokens=tokens,
        vocab_size=100,
        batch_size=2,
        context_window=2,
    )

    try:
        dataset[10]
        assert False, "Should have raised KeyError"
    except KeyError as e:
        assert "idx=10" in str(e)


def test_iterator_functionality():
    tokens = [1, 2, 3, 4, 5, 6]
    dataset = MmappedCausalLMDataset(
        tokens=tokens,
        vocab_size=100,
        batch_size=2,
        context_window=2,
    )

    items = list(dataset)
    assert len(items) == len(dataset)

    first_inputs, first_outputs = items[0]
    np.testing.assert_array_equal(first_inputs, [1, 2])
    np.testing.assert_array_equal(first_outputs, [2, 3])


def test_ordered_batch_retrieval():
    tokens = list(range(1, 21))
    dataset = MmappedCausalLMDataset(
        tokens=tokens,
        vocab_size=100,
        batch_size=3,
        context_window=4,
    )
    dataset.is_batch = True

    with patch("numpy.random.randint") as mock_randint:
        inputs, outputs = dataset._get_batch(0, random=False)

        assert inputs.shape == (3, 4)
        assert outputs.shape == (3, 4)

        np.testing.assert_array_equal(inputs[0], [1, 2, 3, 4])
        np.testing.assert_array_equal(outputs[0], [2, 3, 4, 5])
        np.testing.assert_array_equal(inputs[1], [2, 3, 4, 5])
        np.testing.assert_array_equal(outputs[1], [3, 4, 5, 6])


def test_shuffled_batch_retrieval():
    tokens = list(range(1, 21))
    dataset = MmappedCausalLMDataset(
        tokens=tokens,
        vocab_size=100,
        batch_size=2,
        context_window=3,
    )
    dataset.is_batch = True

    with patch("numpy.random.randint", return_value=np.array([0, 5])):
        inputs, outputs = dataset._get_shuffled_batch(0)

        assert inputs.shape == (2, 3)
        assert outputs.shape == (2, 3)

        np.testing.assert_array_equal(inputs[0], [1, 2, 3])
        np.testing.assert_array_equal(outputs[0], [2, 3, 4])
        np.testing.assert_array_equal(inputs[1], [6, 7, 8])
        np.testing.assert_array_equal(outputs[1], [7, 8, 9])


def test_gpu_device_setting():
    tokens = [1, 2, 3, 4, 5]
    dataset = MmappedCausalLMDataset(
        tokens=tokens,
        vocab_size=100,
        batch_size=2,
        context_window=2,
    )

    dataset.to_gpu(device=1)
    assert dataset.device == 1

    dataset.from_gpu()
    assert dataset.device is None


def test_unbatch_method():
    tokens = [1, 2, 3, 4, 5]
    dataset = MmappedCausalLMDataset(
        tokens=tokens,
        vocab_size=100,
        batch_size=2,
        context_window=2,
    )

    dataset.is_batch = True
    result = dataset.unbatch()

    assert dataset.is_batch == False
    assert result is dataset


def test_shuffle_method():
    tokens = [1, 2, 3, 4, 5]
    dataset = MmappedCausalLMDataset(
        tokens=tokens,
        vocab_size=100,
        batch_size=2,
        context_window=2,
    )

    dataset.shuffle()
    assert hasattr(dataset, "shuffled")
    assert dataset.shuffled == True
