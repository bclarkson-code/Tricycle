def lr_schedule(
    step: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_steps: int,
    total_steps: int,
) -> float:
    """
    Linear decay LR schedule with warmup
    """
    # avoid an off by one error
    step += 1

    if warmup_steps:
        if total_steps < warmup_steps:
            raise ValueError(
                "Cannot have a warmup longer than the total number of steps"
            )
        if step < warmup_steps:
            return (step / warmup_steps) * max_learning_rate

    coef = 1 - ((step - warmup_steps) / total_steps)
    coef *= max_learning_rate - min_learning_rate
    return min_learning_rate + coef
