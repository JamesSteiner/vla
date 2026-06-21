#!/usr/bin/env python
"""Generate docs/smolvla.png — how SmolVLA produces actions (frozen VLM + trained action expert).

    uv run python scripts/make_smolvla_figure.py
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

fig, ax = plt.subplots(figsize=(11, 4.6))
ax.set_xlim(0, 11)
ax.set_ylim(0, 5)
ax.axis("off")


def box(x, y, w, h, text, fc, fontsize=9, weight="normal"):
    ax.add_patch(
        FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.1", fc=fc, ec="#444", lw=1.4)
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, fontweight=weight, color="#111")


def arrow(x1, y1, x2, y2, color="#666", style="-|>", ls="-"):
    ax.add_patch(
        FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=14, lw=1.6, color=color, linestyle=ls)
    )


GREY = "#e7e7e7"
GOLD = "#ffe39a"
LIGHT = "#f4f4f4"

# --- prefix stream (top): observation -> frozen VLM ---
box(0.2, 3.25, 2.5, 1.25, "Observation\n2 cameras +\nrobot state + instruction", LIGHT, fontsize=8.5)
box(3.1, 3.2, 2.7, 1.35, "SmolVLM2-500M backbone\n(FROZEN)\nSigLIP vision + SmolLM2", GREY, fontsize=9, weight="bold")
arrow(2.7, 3.85, 3.1, 3.85)

# --- suffix stream (bottom): noisy actions -> embedded action tokens ---
box(0.2, 0.35, 2.5, 1.1, "Noisy action chunk $x_t$\n+ timestep $t$", LIGHT, fontsize=8.5)
box(3.1, 0.35, 2.7, 1.1, "embed suffix\n(action tokens)", LIGHT, fontsize=9)
arrow(2.7, 0.9, 3.1, 0.9)

# --- the trained action expert, fed by both streams ---
box(6.5, 1.35, 2.2, 2.3, "Action expert\n(TRAINED)\n\ntransformer over\naction tokens,\ncross-attends to\nthe prefix", GOLD, fontsize=8.8, weight="bold")
arrow(5.8, 3.85, 7.0, 3.65, color="#b07d00", ls="--")  # cross-attention from VLM
ax.text(6.35, 3.95, "cross-attention", ha="center", fontsize=8, color="#b07d00", style="italic")
arrow(5.8, 0.9, 7.0, 1.55)  # action tokens into the expert

# --- output head ---
box(9.0, 1.9, 1.85, 1.2, "action_out_proj\n→ velocity →\nintegrate →\n50 × 7 actions", "#cfe0f7", fontsize=8.5)
arrow(8.7, 2.5, 9.0, 2.5)

ax.text(5.5, 4.78, "SmolVLA: a frozen VLM conditions a small, trained action expert", ha="center", fontsize=12, fontweight="bold")
fig.tight_layout()
os.makedirs("docs", exist_ok=True)
fig.savefig("docs/smolvla.png", dpi=150, bbox_inches="tight")
print("saved docs/smolvla.png")
