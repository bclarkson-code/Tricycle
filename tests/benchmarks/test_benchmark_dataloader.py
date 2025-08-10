import pytest

from tricycle.dataset import CausalLMDataset, MmappedCausalLMDataset
from tricycle_datasets.shakespeare import Shakespeare


def load_dataloader(dataloader):
    class SharedConfig:
        vocab_size: int = 1024
        batch_size: int = 32
        context_window: int = 1024
        n_batches: int = int(1e4)
        gpu_id = 0

    config = SharedConfig()

    train_dataset = Shakespeare(config.vocab_size)
    train_dataloader = (
        dataloader(
            tokens=train_dataset.tokens * 1000,
            vocab_size=config.vocab_size,
            batch_size=config.batch_size,
            context_window=config.context_window,
        )
        .batch()
        .shuffle()
        .to_tensor()
    )
    return config, train_dataloader


def iterate_dataloader(train_dataloader, config):
    for step, (inputs, outputs) in enumerate(train_dataloader):
        if step >= config.n_batches:
            break
        assert inputs.to_gpu(config.gpu_id)
        assert outputs.to_gpu(config.gpu_id)


def run_dataloader(dataloader):
    config, dataloader = load_dataloader(dataloader)
    iterate_dataloader(dataloader, config)


@pytest.mark.parametrize(
    "dataloader", [CausalLMDataset, MmappedCausalLMDataset]
)
def test_dataloader(benchmark, dataloader):
    benchmark(run_dataloader, dataloader=dataloader)


@pytest.mark.parametrize(
    "dataloader", [CausalLMDataset, MmappedCausalLMDataset]
)
def test_dataloader_iteration(benchmark, dataloader):
    config, train_dataloader = load_dataloader(dataloader)
    benchmark(
        iterate_dataloader, config=config, train_dataloader=train_dataloader
    )
