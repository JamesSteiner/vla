#!/usr/bin/env python
"""Training entry point — head selected by config. ONE loop, all three conditions.

    uv run python scripts/train.py --config configs/flow.yaml --smoke         # local MPS sanity
    uv run python scripts/train.py --config configs/regression.yaml --smoke    # head sanity
    uv run python scripts/train.py --config configs/flow.yaml                   # full run (CUDA)

Controlled comparison: every head trains the SAME ~100M SmolVLA action
expert; only the OBJECTIVE + inference differ (see vla.heads.expert_objectives):
  * flow       : native flow-matching MSE (policy.forward).
  * regression : deterministic forward + L1.
  * diffusion  : DDPM/DDIM.
The VLM backbone stays frozen (train_expert_only=True). All heads use SmolVLA's
optimizer+scheduler preset (warmup -> cosine decay), same as lerobot-train.
"""

from __future__ import annotations

import argparse
import os

import torch
import yaml

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

from vla.data import make_dataloader, make_libero_splits
from vla.policy.smolvla_wrapper import make_smolvla_processors


def load_config(path: str) -> dict:
    """Load a head config, shallow-merging a `defaults:` base file (head keys win)."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    base_name = cfg.pop("defaults", None)
    if base_name:
        with open(os.path.join(os.path.dirname(path), base_name)) as f:
            base = yaml.safe_load(f)
        base.update(cfg)
        cfg = base
    return cfg


def build_policy(cfg: dict, device: str, resume: str | None = None, resume_subfolder: str | None = None):
    """Load SmolVLA with features ADAPTED to the dataset; VLM frozen, expert trainable.

    resume: continue from an already-finetuned checkpoint (local dir or HF repo) instead of
    starting from smolvla_base — its config already carries the dataset-adapted features.
    """
    ds_meta = LeRobotDatasetMetadata(cfg["dataset_repo_id"])

    if resume:
        path = resume
        if resume_subfolder:
            from huggingface_hub import snapshot_download

            path = os.path.join(snapshot_download(resume, allow_patterns=f"{resume_subfolder}/*"), resume_subfolder)
        policy = SmolVLAPolicy.from_pretrained(path)  # adapted config + partially-trained weights
        policy.config.device = device
        print(f"[train] RESUMED from {path}")
    else:
        pcfg = PreTrainedConfig.from_pretrained(cfg["checkpoint"])
        pcfg.pretrained_path = cfg["checkpoint"]
        pcfg.device = device
        # Clear the checkpoint's own features so make_policy repopulates them from THIS dataset
        # (smolvla_base was pretrained on different camera/state keys than lerobot/libero).
        pcfg.input_features = {}
        pcfg.output_features = {}
        policy = make_policy(pcfg, ds_meta=ds_meta)  # train_expert_only=True => VLM frozen

    policy.to(device)
    pre, post = make_smolvla_processors(policy, dataset_stats=ds_meta.stats)
    return policy, pre, post, ds_meta


def trainable_params(policy):
    return [p for p in policy.parameters() if p.requires_grad]


def compute_loss(policy, cfg: dict, batch):
    """Dispatch the per-objective training loss; all heads train the shared expert."""
    head = cfg["head"]
    if head == "flow":
        loss, _ = policy.forward(batch)  # native flow-matching MSE; masks action_is_pad internally
        return loss
    if head == "regression":
        from vla.heads.expert_objectives import regression_loss

        return regression_loss(policy, batch)
    if head == "diffusion":
        from vla.heads.expert_objectives import diffusion_loss

        return diffusion_loss(policy, batch, cfg.get("diffusion"))
    raise NotImplementedError(f"head={head!r} not wired (expected flow|regression|diffusion)")


def run_smoke(policy, cfg, pre, post, train_ds, steps: int, batch_size: int):
    """Single-batch overfit: the loss should trend down over `steps`."""
    loader = make_dataloader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    batch = pre(next(iter(loader)))  # preprocess once; reuse the SAME batch every step
    params = trainable_params(policy)
    opt = torch.optim.AdamW(params, lr=1e-4)
    print(f"[smoke] head={cfg['head']} trainable={sum(p.numel() for p in params) / 1e6:.1f}M | overfit 1 batch x {steps}")

    policy.train()
    losses = []
    for step in range(steps):
        loss = compute_loss(policy, cfg, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad()
        losses.append(loss.item())
        if step == 0 or (step + 1) % 5 == 0:
            print(f"  step {step + 1:3d}/{steps}  loss={loss.item():.4f}")
    trend = "DOWN (good)" if losses[-1] < losses[0] else "flat/up (investigate)"
    print(f"[smoke] loss {losses[0]:.4f} -> {losses[-1]:.4f}  | trend: {trend}")
    return losses


def maybe_enable_checkpointing(policy, cfg: dict):
    """Optional gradient checkpointing on the VLM backbone (trades compute for VRAM)."""
    if not cfg.get("gradient_checkpointing"):
        return
    try:
        policy.model.vlm_with_expert.vlm.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        print("[train] gradient checkpointing enabled on the VLM")
    except Exception as e:  # noqa: BLE001 — non-fatal; just log and continue
        print(f"[train] gradient checkpointing NOT enabled ({type(e).__name__}: {e})")


def _make_logger(cfg: dict, no_wandb: bool):
    """Return a log(dict, step) fn. W&B if available/enabled, else a no-op."""
    if no_wandb:
        print("[train] W&B disabled (--no-wandb)")
        return lambda d, step: None
    import wandb

    wandb.init(project=cfg["wandb_project"], config=cfg)
    return lambda d, step: wandb.log(d, step=step)


def _push_to_hub(local_dir: str, repo_id: str):
    """Upload a saved checkpoint folder to HF as repo/<folder-name>. Non-fatal on failure.

    Needs write auth — export HF_TOKEN (huggingface_hub reads it) or run `hf auth login`.
    """
    try:
        from huggingface_hub import HfApi

        sub = os.path.basename(local_dir.rstrip("/"))
        api = HfApi()
        api.create_repo(repo_id, exist_ok=True)
        api.upload_folder(folder_path=local_dir, repo_id=repo_id, path_in_repo=sub)
        print(f"[train] pushed -> {repo_id}/{sub}")
    except Exception as e:  # noqa: BLE001 — never let a hub hiccup kill training
        print(f"[train] hub push FAILED (continuing): {type(e).__name__}: {e}")


def _save(policy, pre, post, path: str, hub_repo: str | None = None):
    policy.save_pretrained(path)
    pre.save_pretrained(path)
    post.save_pretrained(path)
    print(f"[train] saved -> {path}")
    if hub_repo:
        _push_to_hub(path, hub_repo)


class _nullctx:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


@torch.no_grad()
def _validate(policy, cfg, pre, val_ds, n_batches: int = 4) -> float:
    policy.eval()
    loader = make_dataloader(
        val_ds, batch_size=cfg.get("batch_size", 64), shuffle=False, num_workers=cfg.get("num_workers", 4)
    )
    total, n = 0.0, 0
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        total += compute_loss(policy, cfg, pre(batch)).item()
        n += 1
    policy.train()
    return total / max(n, 1)


def run_full(policy, pre, post, cfg: dict, train_ds, val_ds, device: str, no_wandb: bool = False):
    """Full training loop (CUDA). Gradient accumulation + SmolVLA optimizer/scheduler preset."""
    import time

    head_name = cfg["head"]
    out = cfg.get("output_dir", f"outputs/{head_name}")
    hub_repo = cfg.get("hub_repo")  # if set, each checkpoint is pushed to HF as it's saved
    save_every = int(cfg.get("save_every", 0))
    batch_size = cfg.get("batch_size", 64)
    grad_accum = max(1, int(cfg.get("grad_accum", 1)))
    loader = make_dataloader(train_ds, batch_size=batch_size, num_workers=cfg.get("num_workers", 4))
    params = trainable_params(policy)
    # SmolVLA's OWN preset (AdamW + warmup→cosine-decay LR schedule), same as lerobot-train.
    # The cosine decay matters: a constant LR with no decay underfits into a timid, under-magnitude policy.
    opt_preset = policy.config.get_optimizer_preset()
    opt = opt_preset.build(params)
    if cfg.get("scheduler_decay_steps"):
        policy.config.scheduler_decay_steps = int(cfg["scheduler_decay_steps"])  # stretch LR decay (diffusion)
    lr_scheduler = policy.config.get_scheduler_preset().build(opt, cfg["max_steps"])
    grad_clip = opt_preset.grad_clip_norm
    use_amp = device == "cuda" and cfg.get("precision") == "bf16"
    maybe_enable_checkpointing(policy, cfg)
    log = _make_logger(cfg, no_wandb)
    print(
        f"[train] head={head_name} steps={cfg['max_steps']} "
        f"eff_batch={batch_size}x{grad_accum}={batch_size * grad_accum} "
        f"trainable={sum(p.numel() for p in params) / 1e6:.1f}M amp={use_amp} device={device}"
    )

    policy.train()
    step, micro, seen = 0, 0, 0
    opt.zero_grad()
    t0 = time.time()
    done = False
    while not done:
        for batch in loader:
            batch = pre(batch)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else _nullctx():
                loss = compute_loss(policy, cfg, batch) / grad_accum
            loss.backward()
            micro += 1
            seen += batch_size
            if micro % grad_accum != 0:
                continue  # keep accumulating; don't step yet

            torch.nn.utils.clip_grad_norm_(params, grad_clip)
            opt.step()
            if lr_scheduler is not None:
                lr_scheduler.step()
            opt.zero_grad()
            log({"train/loss": loss.item() * grad_accum, "train/lr": opt.param_groups[0]["lr"]}, step)
            if step % 20 == 0:
                sps = seen / max(time.time() - t0, 1e-6)
                lr = opt.param_groups[0]["lr"]
                print(f"  step {step:5d}  loss={loss.item() * grad_accum:.4f}  lr={lr:.2e}  {sps:.1f} samples/s")
                log({"perf/samples_per_s": sps}, step)
                t0, seen = time.time(), 0
            if step % cfg.get("val_every", 500) == 0:
                log({"val/loss": _validate(policy, cfg, pre, val_ds)}, step)
            if save_every and step > 0 and step % save_every == 0:
                _save(policy, pre, post, os.path.join(out, f"step_{step}"), hub_repo)
            step += 1
            if step >= cfg["max_steps"]:
                done = True
                break

    _save(policy, pre, post, os.path.join(out, "final"), hub_repo)


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", action="store_true", help="single-batch overfit sanity (local MPS)")
    ap.add_argument("--steps", type=int, default=30, help="smoke steps")
    ap.add_argument("--batch-size", type=int, default=None, help="override config batch_size")
    ap.add_argument("--grad-accum", type=int, default=None, help="override config grad_accum")
    ap.add_argument("--max-steps", type=int, default=None, help="override config max_steps (short pilot)")
    ap.add_argument("--no-wandb", action="store_true", help="disable W&B logging")
    ap.add_argument("--resume", default=None, help="continue from a finetuned checkpoint (local dir or HF repo)")
    ap.add_argument("--resume-subfolder", default=None, help="subfolder within --resume repo (e.g. step_15000)")
    ap.add_argument("--hub-repo", default=None, help="push every checkpoint to this HF repo as it saves (needs HF_TOKEN)")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.grad_accum is not None:
        cfg["grad_accum"] = args.grad_accum
    if args.max_steps is not None:
        cfg["max_steps"] = args.max_steps
    if args.hub_repo:
        cfg["hub_repo"] = args.hub_repo
    device = args.device or _default_device()
    print(f"[train] config={args.config} head={cfg['head']} device={device}")

    policy, pre, post, ds_meta = build_policy(cfg, device, resume=args.resume, resume_subfolder=args.resume_subfolder)
    print(f"[train] policy features in={list(policy.config.input_features)} out={list(policy.config.output_features)}")
    train_ds, val_ds = make_libero_splits(cfg["dataset_repo_id"], chunk_size=policy.config.chunk_size)
    print(f"[train] chunk_size={policy.config.chunk_size}  train_eps={train_ds.num_episodes} val_eps={val_ds.num_episodes}")

    if args.smoke:
        run_smoke(policy, cfg, pre, post, train_ds, args.steps, args.batch_size or 2)
    else:
        run_full(policy, pre, post, cfg, train_ds, val_ds, device, no_wandb=args.no_wandb)


if __name__ == "__main__":
    main()
