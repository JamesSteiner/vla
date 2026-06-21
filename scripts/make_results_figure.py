#!/usr/bin/env python
"""Generate docs/results.png — the headline comparison figure (success / latency / smoothness).

Numbers are the final gate run (10 tasks x 8 trials = 80 episodes per head).
    uv run python scripts/make_results_figure.py
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HEADS = ["flow", "regression", "diffusion"]
COLORS = ["#3b7dd8", "#2ca25f", "#e0772b"]
SUCCESS = [73.8, 70.0, 38.8]
SE = [4.9, 5.1, 5.4]
LATENCY_MS = [615, 286, 618]
SMOOTH = [0.162, 0.103, 0.249]

x = np.arange(3)
fig, axes = plt.subplots(1, 3, figsize=(11, 3.7))


def _bars(ax, vals, title, ylabel, fmt, yerr=None, headroom=0.12):
    ax.bar(x, vals, yerr=yerr, capsize=5, color=COLORS, width=0.62)
    top = max(v + (e or 0) for v, e in zip(vals, yerr or [0] * 3))
    for i, v in enumerate(vals):
        off = (yerr[i] if yerr else 0) + top * 0.03
        ax.text(i, v + off, fmt.format(v), ha="center", va="bottom", fontsize=9)
    ax.set_title(title, fontsize=10.5)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_ylim(0, top * (1 + headroom))
    ax.set_xticks(x)
    ax.set_xticklabels(HEADS)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)


_bars(axes[0], SUCCESS, "Task success  (↑ better)", "success rate (%)", "{:.1f}%", yerr=SE)
_bars(axes[1], LATENCY_MS, "Inference latency  (↓ better)", "ms / action chunk", "{:.0f}")
_bars(axes[2], SMOOTH, "Action smoothness  (↓ smoother)", r"mean $\|a_{t+1}-a_t\|$", "{:.3f}")

fig.suptitle(
    "SmolVLA action-objective comparison on LIBERO-Spatial — same backbone + expert, "
    "only the objective varies (80 episodes/head)",
    fontsize=10.5,
)
fig.tight_layout(rect=(0, 0, 1, 0.93))
os.makedirs("docs", exist_ok=True)
fig.savefig("docs/results.png", dpi=150, bbox_inches="tight")
print("saved docs/results.png")
