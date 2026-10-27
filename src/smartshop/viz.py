"""Chart style shared with the Nectar project, so both portfolios read as one system."""

import matplotlib.pyplot as plt
from nectar.viz import BLUE, GREY, GRID, ORANGE, SURFACE, TEXT, TEXT_2, money, set_style, subtitle

from smartshop.config import FIGURES_DIR

__all__ = [
    "BLUE",
    "GREY",
    "GRID",
    "ORANGE",
    "SURFACE",
    "TEXT",
    "TEXT_2",
    "money",
    "pct",
    "save",
    "set_style",
    "subtitle",
]


def save(fig, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{name}.png")
    plt.close(fig)


def pct(decimals: int = 0):
    """Tick formatter for shares stored as 0..1."""
    from matplotlib.ticker import FuncFormatter

    return FuncFormatter(lambda x, _pos=None: f"{x:.{decimals}%}")
