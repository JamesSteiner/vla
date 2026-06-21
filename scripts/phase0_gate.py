#!/usr/bin/env python
"""Phase 0 GATE — roll out a LIBERO-finetuned SmolVLA in the LIBERO sim, report success.

Purpose: validate the WHOLE eval harness (sim render -> obs mapping -> processor ->
checkpoint inference -> action -> sim step -> success detection) before trusting any result.
No official LIBERO SmolVLA exists, so this uses a community checkpoint as the reference
ceiling. A low score here is ambiguous (mediocre
ckpt vs harness bug), but a non-trivial score proves the harness end to end.

macOS one-time setup:  bash scripts/setup_libero_macos.sh
Run:
    uv run python scripts/phase0_gate.py --tasks 2 --trials 1     # smoke (fast)
    uv run python scripts/phase0_gate.py                          # full: 10 tasks x 5 trials

Notes confirmed against installed LeRobot source:
  * state(8) = cat(eef_pos[3], quat2axisangle(eef_quat)[3], gripper_qpos[2])  (LiberoProcessorStep)
  * images are flipped 180 deg before the policy sees them                     (LiberoProcessorStep)
  * checkpoint cameras: agentview->observation.images.image, eye_in_hand->...wrist_image
  * normalization stats are recovered from the checkpoint safetensors (current LeRobot drops
    the in-policy norm buffers on load), then fed to the SmolVLA processor.
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "cgl")  # macOS-native offscreen GL

import argparse
import time

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

from libero.libero import benchmark
from lerobot.envs.libero import TASK_SUITE_MAX_STEPS, LiberoEnv
from lerobot.utils.constants import ACTION, OBS_STATE
from vla.eval.metrics import action_smoothness
from vla.policy.smolvla_wrapper import make_smolvla_processors

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # isort: skip

GATE_CHECKPOINT = "bicmol/smolvla-libero"
SUITE = "libero_spatial"
AGENTVIEW_CAM = "agentview_image"  # raw LIBERO third-person camera
WRIST_CAM = "robot0_eye_in_hand_image"  # raw LIBERO wrist (eye-in-hand) camera


def derive_camera_mapping(policy) -> dict:
    """Map the LIBERO sim cameras to THIS checkpoint's image-feature keys.

    agentview -> 'image' (the third-person key, common to all); eye_in_hand -> the other
    image feature ('wrist_image' for bicmol, 'image2' for the lerobot/libero checkpoint).
    """
    keys = [k.split(".")[-1] for k in policy.config.image_features]
    if "image" not in keys:
        raise ValueError(f"expected an 'image' camera in checkpoint image_features, got {keys}")
    mapping = {AGENTVIEW_CAM: "image"}
    others = [k for k in keys if k != "image"]
    if others:
        mapping[WRIST_CAM] = others[0]
    return mapping


def recover_stats(checkpoint: str) -> dict:
    """Read in-policy norm buffers (legacy bicmol-style ckpts) -> processor stats.

    Accepts a local dir or a hub repo id. NOTE: the lerobot/libero checkpoints store
    normalization in the processor, not these buffers — use --stats-dataset for those.
    """
    p = (
        os.path.join(checkpoint, "model.safetensors")
        if os.path.isdir(checkpoint)
        else hf_hub_download(checkpoint, "model.safetensors")
    )
    with safe_open(p, framework="pt") as f:
        return {
            OBS_STATE: {
                "mean": f.get_tensor("normalize_inputs.buffer_observation_state.mean"),
                "std": f.get_tensor("normalize_inputs.buffer_observation_state.std"),
            },
            ACTION: {
                "mean": f.get_tensor("unnormalize_outputs.buffer_action.mean"),
                "std": f.get_tensor("unnormalize_outputs.buffer_action.std"),
            },
        }


def quat2axisangle(quat: torch.Tensor) -> torch.Tensor:
    """(x, y, z, w) quaternion -> axis-angle (3,). Mirrors LiberoProcessorStep._quat2axisangle."""
    q = quat.to(torch.float32)
    w = q[3].clamp(-1.0, 1.0)
    den = torch.sqrt(1.0 - w * w)
    if den < 1e-8:
        return torch.zeros(3, dtype=torch.float32)
    return (q[:3] / den) * (2.0 * torch.acos(w))


def build_obs(raw: dict, task_desc: str) -> dict:
    """LiberoEnv 'pixels_agent_pos' obs -> a SmolVLA input batch (B=1), pre-processor steps applied."""
    obs: dict = {}
    for cam_key, img in raw["pixels"].items():  # cam_key in {"image", "wrist_image"}
        t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0  # [C,H,W]
        t = torch.flip(t, dims=[1, 2]).unsqueeze(0)  # 180-deg flip + batch -> [1,C,H,W]
        obs[f"observation.images.{cam_key}"] = t

    rs = raw["robot_state"]
    eef_pos = torch.as_tensor(rs["eef"]["pos"], dtype=torch.float32)  # (3,)
    eef_quat = torch.as_tensor(rs["eef"]["quat"], dtype=torch.float32)  # (4,) xyzw
    gripper = torch.as_tensor(rs["gripper"]["qpos"], dtype=torch.float32)  # (2,)
    state = torch.cat([eef_pos, quat2axisangle(eef_quat), gripper]).unsqueeze(0)  # [1,8]
    obs[OBS_STATE] = state
    obs["task"] = [task_desc]
    return obs


def _chunk_sampler(policy, head_name: str):
    """For the custom heads, return a fn(batch) -> normalized chunk [1, chunk, action_dim]."""
    if head_name == "regression":
        from vla.heads.expert_objectives import regression_sample

        return regression_sample
    if head_name == "diffusion":
        from vla.heads.expert_objectives import diffusion_sample

        return lambda p, b: diffusion_sample(p, b, None)  # default DDIM (10 steps), matches diffusion.yaml
    return None  # flow uses native select_action


def run_episode(policy, pre, post, env, task_desc, max_steps, head_name: str = "flow") -> tuple[bool, float, int, float]:
    """One episode. Returns (success, total_inference_seconds, n_inferences).

    flow uses SmolVLA's native sampling + action queue; regression/diffusion sample a chunk
    via the shared expert (one objective-specific forward) and execute n_action_steps of it.
    """
    sampler = _chunk_sampler(policy, head_name)
    policy.reset()
    raw, _ = env.reset()
    success = False
    infer_s, n_infer = 0.0, 0
    queue: list = []
    acts: list = []
    for _ in range(max_steps):
        batch = pre(build_obs(raw, task_desc))
        if sampler is not None:  # regression / diffusion
            if not queue:
                t0 = time.time()
                chunk = sampler(policy, batch)  # [1, chunk, adim] normalized
                infer_s += time.time() - t0
                n_infer += 1
                chunk = np.asarray(post(chunk).squeeze(0).to("cpu"))[:, :7].astype(np.float32)  # [chunk, 7]
                queue = list(chunk[: policy.config.n_action_steps])
            a = queue.pop(0)
        else:  # flow — native queue
            queue_was_empty = len(policy._queues[ACTION]) == 0
            t0 = time.time()
            with torch.no_grad():
                action = policy.select_action(batch)  # [1,7], normalized
            if queue_was_empty:  # only count real forward passes, not queue pops
                infer_s += time.time() - t0
                n_infer += 1
            a = np.asarray(post(action).squeeze(0).to("cpu")).reshape(-1)[:7].astype(np.float32)
        acts.append(a)
        raw, _, terminated, truncated, info = env.step(a)
        if info.get("is_success"):
            success = True
        if terminated or truncated:
            break
    sm = action_smoothness(torch.tensor(np.asarray(acts), dtype=torch.float32)) if len(acts) >= 2 else 0.0
    return success, infer_s, n_infer, sm


def main():
    ap = argparse.ArgumentParser(description="Phase 0 LIBERO gate for a finetuned SmolVLA")
    ap.add_argument("--checkpoint", default=GATE_CHECKPOINT)
    ap.add_argument("--subfolder", default=None, help="load this subfolder from the repo (e.g. step_15000)")
    ap.add_argument(
        "--stats-dataset",
        default=None,
        help="build norm stats from this dataset (e.g. lerobot/libero) instead of recovering from buffers",
    )
    ap.add_argument("--suite", default=SUITE)
    ap.add_argument("--tasks", type=int, default=10, help="number of tasks (capped at suite size)")
    ap.add_argument("--trials", type=int, default=5, help="rollouts per task (each a different init state)")
    ap.add_argument("--max-steps", type=int, default=None, help="override episode length")
    ap.add_argument("--head", default="flow", choices=["flow", "regression", "diffusion"], help="inference objective")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()

    ckpt = args.checkpoint
    if args.subfolder:
        from huggingface_hub import snapshot_download

        root = snapshot_download(args.checkpoint, allow_patterns=f"{args.subfolder}/*")
        ckpt = os.path.join(root, args.subfolder)

    print(f"[gate] checkpoint={ckpt} head={args.head} suite={args.suite} device={args.device}")
    print("[gate] loading policy...")
    policy = SmolVLAPolicy.from_pretrained(ckpt)
    policy.config.device = args.device
    policy.to(args.device)
    policy.eval()

    if args.stats_dataset:
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

        stats = LeRobotDatasetMetadata(args.stats_dataset).stats
        print(f"[gate] norm stats from dataset {args.stats_dataset}")
    else:
        stats = recover_stats(ckpt)
        print("[gate] norm stats recovered from checkpoint buffers")
    pre, post = make_smolvla_processors(policy, dataset_stats=stats)
    camera_mapping = derive_camera_mapping(policy)
    print(f"[gate] camera mapping: {camera_mapping}")

    suite = benchmark.get_benchmark_dict()[args.suite]()
    n_tasks = min(args.tasks, suite.n_tasks)
    max_steps = args.max_steps or TASK_SUITE_MAX_STEPS.get(args.suite, 500)
    print(f"[gate] {n_tasks} tasks x {args.trials} trials, max_steps={max_steps}\n")

    total_succ, total_eps, lat_ms, smooth = 0, 0, [], []
    t_start = time.time()
    for tid in range(n_tasks):
        env = LiberoEnv(
            task_suite=suite,
            task_id=tid,
            task_suite_name=args.suite,
            obs_type="pixels_agent_pos",
            camera_name_mapping=camera_mapping,
            observation_height=256,
            observation_width=256,
        )
        task_desc = env.task_description
        succ = 0
        for trial in range(args.trials):
            ok, infer_s, n_infer, sm = run_episode(policy, pre, post, env, task_desc, max_steps, head_name=args.head)
            succ += int(ok)
            smooth.append(sm)
            if n_infer:
                lat_ms.append(1000.0 * infer_s / n_infer)
            print(f"  task {tid} trial {trial}: {'SUCCESS' if ok else 'fail   '}  ({n_infer} infers)")
        env.close()
        total_succ += succ
        total_eps += args.trials
        print(f"  -> task {tid}: {succ}/{args.trials}  | {task_desc}\n")

    p = total_succ / max(total_eps, 1)
    rate = 100.0 * p
    se = 100.0 * (p * (1 - p) / max(total_eps, 1)) ** 0.5  # binomial standard error
    mean_lat = float(np.mean(lat_ms)) if lat_ms else float("nan")
    mean_sm = float(np.mean(smooth)) if smooth else float("nan")
    print("=" * 60)
    print(f"[gate] head={args.head}  success: {total_succ}/{total_eps} = {rate:.1f}% (±{se:.1f} SE)")
    print(f"[gate] mean inference latency: {mean_lat:.0f} ms/chunk  ({args.device})")
    print(f"[gate] mean action smoothness: {mean_sm:.4f}  (mean ||a_t+1 - a_t||; lower = smoother)")
    print(f"[gate] wall time: {time.time() - t_start:.0f}s")
    print("[gate] GATE PASSED (non-trivial success)" if rate > 0 else "[gate] GATE FAILED (0% — investigate harness/ckpt)")


if __name__ == "__main__":
    main()
