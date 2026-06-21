#!/usr/bin/env python
"""Render a LIBERO rollout of a trained policy to docs/rollout.gif (agentview).

Rolls out until the policy succeeds on a task, then saves that episode.
    uv run python scripts/make_rollout_gif.py
"""

import os
import sys

os.environ.setdefault("MUJOCO_GL", "cgl")
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))  # import phase0_gate helpers

import numpy as np  # noqa: E402
import torch  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402
from PIL import Image  # noqa: E402

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata  # noqa: E402
from lerobot.envs.libero import TASK_SUITE_MAX_STEPS, LiberoEnv  # noqa: E402
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # noqa: E402
from libero.libero import benchmark  # noqa: E402

import phase0_gate as G  # noqa: E402
from vla.policy.smolvla_wrapper import make_smolvla_processors  # noqa: E402

CKPT = "james-steiner/smolvla-libero-flow-v3"
SUBFOLDER = "step_30000"

ckpt = os.path.join(snapshot_download(CKPT, allow_patterns=f"{SUBFOLDER}/*"), SUBFOLDER)
policy = SmolVLAPolicy.from_pretrained(ckpt)
policy.config.device = "mps"
policy.to("mps")
policy.eval()
pre, post = make_smolvla_processors(policy, dataset_stats=LeRobotDatasetMetadata("lerobot/libero").stats)
cam = G.derive_camera_mapping(policy)
suite = benchmark.get_benchmark_dict()["libero_spatial"]()
max_steps = TASK_SUITE_MAX_STEPS["libero_spatial"]


def rollout(task_id):
    env = LiberoEnv(
        task_suite=suite, task_id=task_id, task_suite_name="libero_spatial",
        obs_type="pixels_agent_pos", camera_name_mapping=cam, observation_height=256, observation_width=256,
    )
    policy.reset()
    raw, _ = env.reset()
    frames, success = [], False
    for _ in range(max_steps):
        frames.append(np.asarray(raw["pixels"]["image"]).astype(np.uint8))  # agentview (human-upright)
        with torch.no_grad():
            a = post(policy.select_action(pre(G.build_obs(raw, env.task_description))))
        a = np.asarray(a.squeeze(0).cpu()).reshape(-1)[:7].astype(np.float32)
        raw, _, terminated, truncated, info = env.step(a)
        if info.get("is_success"):
            success = True
        if terminated or truncated:
            break
    desc = env.task_description
    env.close()
    return frames, success, desc


for tid in range(suite.n_tasks):
    frames, success, desc = rollout(tid)
    print(f"  task {tid}: {'SUCCESS' if success else 'fail'} ({len(frames)} steps) | {desc}")
    if success:
        frames = frames[::3]  # subsample for a ~2-3s clip
        pil = [Image.fromarray(f) for f in frames]
        os.makedirs("docs", exist_ok=True)
        pil[0].save("docs/rollout.gif", save_all=True, append_images=pil[1:], duration=80, loop=0)
        print(f"saved docs/rollout.gif ({len(pil)} frames) | task: {desc}")
        break
