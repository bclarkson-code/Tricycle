import math

from matplotlib import pyplot as plt


def linear_schedule(
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


class CosineSchedule:
    def __init__(
        self,
        max_learning_rate: float,
        min_learning_rate: float,
        total_steps: int,
        warmup_steps: int = 0,
    ):
        self.max_learning_rate = max_learning_rate
        self.min_learning_rate = min_learning_rate
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps

        if self.total_steps < self.warmup_steps:
            raise ValueError(
                "Cannot have a warmup longer than the total number of steps"
            )

        self.n_steps = total_steps - warmup_steps
        self.coef = (self.max_learning_rate - self.min_learning_rate) * 0.5

    def step(
        self,
        step: int,
    ) -> float:
        """
        Cosine decay schedule with warmup
        """

        if step < self.warmup_steps:
            return (step / self.warmup_steps) * self.max_learning_rate

        if self.warmup_steps <= step < self.total_steps:
            idx = math.pi * (step - self.warmup_steps) / self.n_steps

            return self.min_learning_rate + self.coef * (math.cos(idx) + 1)

        return self.min_learning_rate

    def __call__(self, step: int) -> float:
        return self.step(step)


if __name__ == "__main__":
    x = [i for i in range(5000)]
    schedule = CosineSchedule(
        max_learning_rate=6e-4,
        min_learning_rate=0,
        total_steps=5000,
        warmup_steps=100,
    )
    y = [schedule(i) for i in x]
    plt.plot(x, y)
    plt.savefig("out.png")
