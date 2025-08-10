def test_train_shakespeare(benchmark):
    benchmark(train_shakespeare)


def test_train_shakespeare_update_only(benchmark):
    from tricycle import GPU_ENABLED
    from tricycle.configs import ShakespeareConfig
    from tricycle.dataset import MmappedCausalLMDataset
    from tricycle.loss import CrossEntropy
    from tricycle.models import GPT
    from tricycle.optimisers import AdamW
    from tricycle_datasets.shakespeare import Shakespeare

    config = ShakespeareConfig()
    config.steps = 100

    model = GPT(config)

    tokens = Shakespeare(vocab_size=config.vocab_size)
    dataset = (
        MmappedCausalLMDataset(
            tokens=tokens,
            vocab_size=config.vocab_size,
            batch_size=config.batch_size,
            context_window=config.context_window,
        )
        .batch()
        .shuffle()
        .to_tensor()
    )
    loss_fn = CrossEntropy()
    optimiser = AdamW(
        learning_rate=config.max_learning_rate,
        weight_decay=config.weight_decay,
        betas=(config.beta1, config.beta2),
    )

    if GPU_ENABLED:
        dataset = dataset.to_gpu()
        model.to_gpu()

    benchmark(
        train_shakespeare_no_startup,
        config=config,
        optimiser=optimiser,
        dataset=dataset,
        model=model,
        loss_fn=loss_fn,
    )


def train_shakespeare():
    from tricycle import GPU_ENABLED
    from tricycle.configs import ShakespeareConfig
    from tricycle.dataset import CausalLMDataset
    from tricycle.loss import CrossEntropy
    from tricycle.models import GPT
    from tricycle.optimisers import AdamW
    from tricycle_datasets.shakespeare import Shakespeare

    config = ShakespeareConfig()
    config.steps = 100

    model = GPT(config)

    tokens = Shakespeare(vocab_size=config.vocab_size)
    dataset = (
        CausalLMDataset(
            tokens=tokens,
            vocab_size=config.vocab_size,
            batch_size=config.batch_size,
            context_window=config.context_window,
        )
        .batch()
        .shuffle()
        .to_tensor()
    )
    loss_fn = CrossEntropy()
    optimiser = AdamW(
        learning_rate=config.max_learning_rate,
        weight_decay=config.weight_decay,
        betas=(config.beta1, config.beta2),
    )

    if GPU_ENABLED:
        dataset = dataset.to_gpu()
        model.to_gpu()

    for _ in range(config.steps):
        optimiser.step()
        inputs, outputs = next(dataset)

        logits = model(inputs)
        loss = loss_fn(outputs, logits)
        loss.backward()

        model.update(optimiser)


def train_shakespeare_no_startup(config, optimiser, dataset, model, loss_fn):
    for _ in range(config.steps):
        optimiser.step()
        inputs, outputs = next(dataset)

        logits = model(inputs)
        loss = loss_fn(outputs, logits)
        loss.backward()

        model.update(optimiser)
