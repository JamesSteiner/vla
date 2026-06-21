#!/usr/bin/env python
"""Generate docs/architecture.png — the controlled-comparison design (one shared expert).

    uv run python scripts/make_architecture_figure.py
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

fig, ax = plt.subplots(figsize=(11, 4.3))
ax.set_xlim(0, 11)
ax.set_ylim(0, 5)
ax.axis("off")


def box(x, y, w, h, text, fc, fontsize=9, weight="normal"):
    ax.add_patch(
        FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.12", fc=fc, ec="#444", lw=1.4)
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, fontweight=weight, color="#111")


def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=15, lw=1.6, color="#666"))


ax.text(5.5, 4.75, "Same everything — only the objective varies", ha="center", fontsize=13, fontweight="bold")

# pipeline (left to right)
box(0.15, 2.0, 1.95, 1.05, "Observation\n2 cameras + state\n+ language", "#f4f4f4")
box(2.55, 1.9, 1.95, 1.25, "SmolVLM backbone\n(FROZEN)", "#e7e7e7", fontsize=10, weight="bold")
box(5.0, 2.15, 1.2, 0.85, "prefix\n(conditioning)", "#f4f4f4")
box(6.65, 1.55, 2.05, 1.95, "Action expert\ntransformer +\ncross-attention\n\nTRAINED —\nsame for all 3", "#ffe39a", fontsize=9, weight="bold")
for x1, x2 in [(2.1, 2.55), (4.5, 5.0), (6.2, 6.65)]:
    arrow(x1, 2.5, x2, 2.5)

# three objectives branching off the shared expert
box(9.05, 3.5, 1.85, 0.95, "Flow matching\npredict velocity · MSE", "#cfe0f7", fontsize=8.5)
box(9.05, 2.02, 1.85, 0.95, "Regression\npredict action · L1", "#cdeed9", fontsize=8.5)
box(9.05, 0.55, 1.85, 0.95, "Diffusion\npredict noise ε · MSE", "#f7ddc2", fontsize=8.5)
arrow(8.7, 2.9, 9.05, 3.95)
arrow(8.7, 2.5, 9.05, 2.5)
arrow(8.7, 2.1, 9.05, 1.05)

fig.tight_layout()
os.makedirs("docs", exist_ok=True)
fig.savefig("docs/architecture.png", dpi=150, bbox_inches="tight")
print("saved docs/architecture.png")
