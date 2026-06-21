#!/usr/bin/env python
"""Generate docs/objectives.png — how each objective turns noise into an action chunk.

Same shared expert, three samplers: regression takes one deterministic pass; diffusion
denoises over ~10 stochastic steps; flow integrates a straight path over ~10 steps.

    uv run python scripts/make_objectives_figure.py
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

rng = np.random.default_rng(3)

BLUE = "#3b7dd8"    # flow
GREEN = "#2ca25f"   # regression
ORANGE = "#e0772b"  # diffusion

fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
for ax in axes:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def action_star(ax, color):
    ax.scatter([0.9], [0.5], marker="*", s=460, color=color, zorder=6, edgecolor="#222", linewidth=0.6)
    ax.text(0.9, 0.32, "action", ha="center", va="top", fontsize=9, color="#222")


def noise_cloud(ax):
    xs = 0.1 + rng.normal(0, 0.022, 60)
    ys = 0.5 + rng.normal(0, 0.13, 60)
    ax.scatter(xs, ys, s=8, color="#9aa0a6", alpha=0.5, zorder=1)
    ax.text(0.1, 0.12, "noise", ha="center", va="top", fontsize=9, color="#555")


def step_label(ax, text, color):
    ax.text(0.5, 0.9, text, ha="center", fontsize=9.5, color=color, style="italic")


def caption(ax, text):
    ax.text(0.5, 0.03, text, ha="center", fontsize=9, color="#444")


# --- Regression: fixed query -> action in one pass -----------------------------------
ax = axes[0]
ax.set_title("Regression", color=GREEN, fontsize=12, fontweight="bold", pad=10)
ax.add_patch(Rectangle((0.085, 0.47), 0.03, 0.06, color="#9aa0a6", zorder=2))
ax.text(0.1, 0.12, "fixed query", ha="center", va="top", fontsize=9, color="#555")
ax.add_patch(FancyArrowPatch((0.13, 0.5), (0.855, 0.5), arrowstyle="-|>", mutation_scale=20, lw=2.6, color=GREEN, zorder=3))
action_star(ax, GREEN)
step_label(ax, "one forward pass", GREEN)
caption(ax, "deterministic · 1 step")

# --- Diffusion: noise sample -> action over ~10 stochastic steps ----------------------
ax = axes[1]
ax.set_title("Diffusion (DDPM)", color=ORANGE, fontsize=12, fontweight="bold", pad=10)
noise_cloud(ax)
n = 10
xs = np.linspace(0.12, 0.86, n + 1)
ys = np.linspace(0.64, 0.5, n + 1) + rng.normal(0, 1, n + 1) * np.linspace(0.08, 0.0, n + 1)
ys[-1] = 0.5
ax.plot(xs, ys, "-", color=ORANGE, lw=2.2, zorder=3)
ax.scatter(xs[:-1], ys[:-1], s=22, color=ORANGE, zorder=4)
action_star(ax, ORANGE)
step_label(ax, "~10 stochastic steps", ORANGE)
caption(ax, "denoise · curved, stochastic")

# --- Flow matching: noise sample -> action along a straight path ----------------------
ax = axes[2]
ax.set_title("Flow matching", color=BLUE, fontsize=12, fontweight="bold", pad=10)
noise_cloud(ax)
xs = np.linspace(0.12, 0.86, 11)
ys = np.linspace(0.64, 0.5, 11)
ax.plot(xs, ys, "-", color=BLUE, lw=2.2, zorder=3)
ax.scatter(xs[:-1], ys[:-1], s=22, color=BLUE, zorder=4)
action_star(ax, BLUE)
step_label(ax, "~10 ODE steps", BLUE)
caption(ax, "integrate · straight path")

fig.suptitle("From noise to an action chunk — same expert, three samplers", fontsize=12.5, fontweight="bold", y=1.0)
fig.tight_layout(rect=(0, 0, 1, 0.96))
os.makedirs("docs", exist_ok=True)
fig.savefig("docs/objectives.png", dpi=150, bbox_inches="tight")
print("saved docs/objectives.png")
