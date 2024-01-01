"""
Some utils that help with running experiments (I couldn't find anywhere else
for them to go)
"""

from matplotlib import pyplot as plt
from matplotlib.widgets import Slider


def smooth(values: list[float], factor=0.99):
    """
    Apply exponential smoothing to a list of values
    """
    assert 0 <= factor <= 1

    smoothed = values[0]
    for val in values[1:]:
        smoothed = factor * smoothed + (1 - factor) * val
        yield smoothed


def plot_loss(losses):
    """
    Plot a loss curve with a slider to control smoothing
    """
    factor = 0.99

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_ylim(min(losses) * 0.95, max(losses) * 1.05)

    line = ax.plot(list(smooth(losses, factor=factor)))
    fig.subplots_adjust(left=0.25, bottom=0.25)

    slider_ax = fig.add_axes((0.25, 0.1, 0.65, 0.03))
    slider = Slider(
        ax=slider_ax,
        label="Smoothing",
        valmin=0,
        valmax=1,
        valinit=factor,
    )

    def update(factor: float):
        line[0].set_ydata(list(smooth(losses, factor=factor)))
        plt.gcf().canvas.draw()

    slider.on_changed(update)

    plt.show()
