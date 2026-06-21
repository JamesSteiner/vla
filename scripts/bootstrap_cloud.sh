#!/usr/bin/env bash
# Cloud bootstrap for the SmolVLA flow fine-tune on a fresh Ubuntu + NVIDIA CUDA box
# (RunPod / Lambda / Vast / any SSH GPU instance). Assumes the NVIDIA driver is present
# (every GPU cloud image has it). It clones, installs, verifies CUDA, and launches training
# under nohup (survives SSH disconnect). Eval/the LIBERO gate runs on the Mac, not here.
#
# Recommended host: RunPod community-cloud RTX 4090 (24GB), official PyTorch template, ~40GB disk.
#
# Usage (SSH into the pod, then):
#   export WANDB_API_KEY=...     # optional; omit -> trains with --no-wandb
#   export HF_TOKEN=...          # optional; only needed to push the checkpoint at the end
#   curl -fsSL https://raw.githubusercontent.com/JamesSteiner/vla/main/scripts/bootstrap_cloud.sh | bash
#   # ...or: git clone the repo first and run: bash scripts/bootstrap_cloud.sh
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/JamesSteiner/vla.git}"
CONFIG="${CONFIG:-configs/flow.yaml}"
BATCH="${BATCH:-32}"          # 24GB fits ~32; if CUDA OOM, drop to 16 (raise GRAD_ACCUM to 4)
GRAD_ACCUM="${GRAD_ACCUM:-2}" # effective batch = BATCH * GRAD_ACCUM (default 64)

echo "==> GPU / driver check"
nvidia-smi || { echo "ERROR: no NVIDIA GPU/driver visible"; exit 1; }

echo "==> install uv"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

echo "==> clone + sync"
if [ ! -d vla ]; then git clone "$REPO_URL" vla; fi
cd vla
git pull --ff-only || true
uv sync

echo "==> verify CUDA torch (Linux PyPI torch wheels bundle CUDA by default)"
uv run python -c "import torch; assert torch.cuda.is_available(), 'CUDA torch NOT available'; print('CUDA OK:', torch.cuda.get_device_name(0))"

echo "==> W&B"
WANDB_FLAG=""
if [ -n "${WANDB_API_KEY:-}" ]; then
  uv run wandb login "$WANDB_API_KEY" >/dev/null 2>&1 && echo "  W&B logged in"
else
  echo "  no WANDB_API_KEY -> training with --no-wandb"
  WANDB_FLAG="--no-wandb"
fi

RESUME_FLAG=""
[ -n "${RESUME:-}" ] && RESUME_FLAG="--resume $RESUME"
[ -n "${RESUME_SUBFOLDER:-}" ] && RESUME_FLAG="$RESUME_FLAG --resume-subfolder $RESUME_SUBFOLDER"

HUB_FLAG=""
if [ -n "${HUB_REPO:-}" ]; then
  HUB_FLAG="--hub-repo $HUB_REPO"
  echo "  auto-push checkpoints -> $HUB_REPO"
  [ -z "${HF_TOKEN:-}" ] && echo "  WARNING: HUB_REPO set but HF_TOKEN not exported — auto-push will fail (run: uv run hf auth login)"
fi

echo "==> launch ($CONFIG) under nohup (survives disconnect) ${RESUME_FLAG:+[resume $RESUME/$RESUME_SUBFOLDER]}"
nohup uv run python scripts/train.py \
  --config "$CONFIG" --batch-size "$BATCH" --grad-accum "$GRAD_ACCUM" $RESUME_FLAG $HUB_FLAG $WANDB_FLAG \
  > train.log 2>&1 &
echo $! > train.pid
echo
echo "  started PID $(cat train.pid)"
echo "  watch : tail -f $(pwd)/train.log     (look for 'CUDA OK', falling val/loss, samples/s)"
echo "  stop  : kill \$(cat $(pwd)/train.pid)   (checkpoints in outputs/<head>/step_* — auto-pushed if HUB_REPO set)"
echo
echo "If NOT auto-pushing, upload manually as checkpoints land:"
echo "  uv run hf upload <hf-user>/<repo> outputs/<head>/step_NNNNN step_NNNNN"
echo "Eval on the Mac:  uv run python scripts/phase0_gate.py --checkpoint <hf-user>/<repo> --subfolder step_NNNNN --head <head> --stats-dataset lerobot/libero"
