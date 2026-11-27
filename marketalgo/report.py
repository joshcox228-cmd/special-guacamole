"""Text table and chart for a finished backtest."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PCT = {"CAGR", "Volatility", "Max drawdown", "Worst month", "Winning months", "Turnover / yr"}

# Fixed colour per strategy so a line keeps its colour in every panel.
COLORS = {
    "ML top-5": "#257de0",
    "Core + momentum rule": "#f47230",
    "Equal weight (all ETFs)": "#00b894",
    "SPY buy & hold": "#f2a600",
}
INK, MUTED, GRID, AXIS = "#191919", "#666666", "#e7e7e7", "#c2c2c2"


def metrics_table(rows: dict[str, dict]) -> str:
    table = pd.DataFrame(rows).T
    for col in table.columns:
        table[col] = table[col].map((lambda v: f"{v:.1%}") if col in PCT else (lambda v: f"{v:.2f}"))
    return table.to_string()


def _style(ax):
    ax.set_facecolor("white")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED, length=0, labelsize=9)


def plot(results: dict[str, pd.DataFrame], path: Path, main: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(11, 7.5), sharex=True, height_ratios=[2.2, 1], facecolor="white")

    ends = []
    for name, res in results.items():
        emphasis = name == main
        top.plot(res.index, res["equity"], color=COLORS[name], label=name,
                 linewidth=2.2 if emphasis else 1.6, zorder=3 if emphasis else 2)
        ends.append((res["equity"].iloc[-1], name))
    top.set_yscale("log")
    top.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:g}"))
    top.minorticks_off()
    top.set_title("Growth of $1, after trading costs (log scale)",
                  loc="left", color=INK, fontsize=12, fontweight="bold")
    top.legend(frameon=False, fontsize=9, labelcolor=MUTED, loc="upper left")
    _style(top)

    # Label line ends directly, nudging apart any that would collide.
    import numpy as np
    last = results[next(iter(results))].index[-1]
    ends.sort()
    placed = []
    for value, name in ends:
        y = np.log(value)
        if placed and y - placed[-1] < 0.09:
            y = placed[-1] + 0.09
        placed.append(y)
        top.annotate(f"${value:.2f}  {name.split()[0]}", xy=(last, np.exp(y)), xytext=(6, 0),
                     textcoords="offset points", va="center", fontsize=9, color=INK)

    for name in ("SPY buy & hold", main):
        eq = results[name]["equity"]
        bottom.plot(eq.index, eq / eq.cummax() - 1, color=COLORS[name], linewidth=1.5, label=name)
    bottom.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    bottom.set_title("Drawdown: fall from the previous peak",
                     loc="left", color=INK, fontsize=12, fontweight="bold")
    bottom.legend(frameon=False, fontsize=9, labelcolor=MUTED, loc="lower left")
    _style(bottom)

    fig.tight_layout()
    fig.subplots_adjust(right=0.86)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
