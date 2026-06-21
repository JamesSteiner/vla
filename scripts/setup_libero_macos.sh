#!/usr/bin/env bash
# One-time LIBERO sim setup for macOS (Apple Silicon).
#
# Why this is non-standard: lerobot[libero] gates `hf-libero` to Linux, and a naive
# `pip install hf-libero` fails on macOS because robomimic -> egl-probe can't build a
# cmake/EGL C-extension. robomimic/egl-probe are NOT needed to RENDER, so we install
# the real engine (robosuite + mujoco + bddl) and the libero package with --no-deps.
# MuJoCo offscreen rendering works on macOS via MUJOCO_GL=cgl.
set -euo pipefail

echo "==> installing robosuite + mujoco + bddl + small deps (skip robomimic/egl-probe)"
uv pip install "robosuite==1.4.0" "mujoco==3.9.0" bddl easydict matplotlib

echo "==> installing the libero package itself (no deps)"
uv pip install "hf-libero>=0.1.3,<0.2.0" --no-deps

echo "==> writing ~/.libero/config.yaml (skips libero's first-run interactive prompt)"
uv run python - <<'PY'
import os, yaml
import libero  # noqa: F401 -- import only to locate the installed package dir
pkg = os.path.join(os.path.dirname(libero.__file__), "libero")  # .../libero/libero
cfg = {
    "benchmark_root": pkg,
    "bddl_files": os.path.join(pkg, "bddl_files"),
    "init_states": os.path.join(pkg, "init_files"),
    "datasets": os.path.join(os.path.dirname(pkg), "datasets"),
    "assets": os.path.join(pkg, "assets"),
}
cfgdir = os.path.expanduser("~/.libero"); os.makedirs(cfgdir, exist_ok=True)
with open(os.path.join(cfgdir, "config.yaml"), "w") as f:
    yaml.dump(cfg, f)
print("wrote", os.path.join(cfgdir, "config.yaml"))
PY

echo "==> done. LIBERO assets (~586 files) auto-download to ~/.cache/libero on first env build."
echo "    Test:  MUJOCO_GL=cgl uv run python scripts/phase0_gate.py --tasks 2 --trials 1"
