"""LIBERO as a LeRobotDataset: action chunking + an episode-level train/val split.

Wired against `lerobot/libero` (codebase_version v3.0):
  observation.images.image   256x256x3   base camera
  observation.images.image2  256x256x3   second camera
  observation.state          dim 8
  action                     dim 7
  task                       language instruction (str)

The action chunk is produced by LeRobot's `delta_timestamps`: it requests the
actions at relative frame offsets 0..chunk_size-1, which turns `action` from
[7] into [chunk_size, 7]. Chunks that run past an episode boundary are padded and
LeRobot returns a companion `action_is_pad` boolean mask — apply it in the loss so
padded steps don't contribute. Images/state are left as the single current frame
(SmolVLA conditions on the current observation only).

Image preprocessing (resize, normalize) is handled by the SmolVLA policy/processor,
so no image_transforms are set here.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

LIBERO_REPO_ID = "lerobot/libero"
VAL_FRACTION = 0.10

# Feature keys in lerobot/libero.
IMAGE_KEY = "observation.images.image"
IMAGE2_KEY = "observation.images.image2"
STATE_KEY = "observation.state"
ACTION_KEY = "action"
ACTION_PAD_KEY = "action_is_pad"  # boolean mask LeRobot adds when chunks are padded
TASK_KEY = "task"


def _action_chunk_delta_timestamps(fps: float, chunk_size: int) -> dict[str, list[float]]:
    """Load action[t .. t+chunk_size-1] as one [chunk_size, action_dim] tensor."""
    return {ACTION_KEY: [i / fps for i in range(chunk_size)]}


def _split_episodes(total_episodes: int, val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    """Deterministic episode-level split (whole episodes held out, never frames)."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(total_episodes, generator=g).tolist()
    n_val = max(1, int(total_episodes * val_fraction))
    val_eps = sorted(perm[:n_val])
    train_eps = sorted(perm[n_val:])
    return train_eps, val_eps


def make_libero_splits(
    repo_id: str = LIBERO_REPO_ID,
    chunk_size: int = 50,
    val_fraction: float = VAL_FRACTION,
    seed: int = 0,
) -> tuple[LeRobotDataset, LeRobotDataset]:
    """Build train/val LeRobotDatasets that yield (current obs, future action chunk).

    `chunk_size` MUST equal the policy's action chunk (read SmolVLAConfig.chunk_size
    from your loaded policy and pass it in — the default here is SmolVLA's typical 50).
    """
    meta = LeRobotDatasetMetadata(repo_id)  # cheap: metadata only, no frame download
    fps = meta.fps
    train_eps, val_eps = _split_episodes(meta.total_episodes, val_fraction, seed)
    delta = _action_chunk_delta_timestamps(fps, chunk_size)

    train = LeRobotDataset(repo_id, episodes=train_eps, delta_timestamps=delta)
    val = LeRobotDataset(repo_id, episodes=val_eps, delta_timestamps=delta)
    return train, val


def make_dataloader(
    dataset: LeRobotDataset,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 4,
) -> DataLoader:
    """Plain torch DataLoader. Default collate handles tensors; `task` stays a list[str]."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=shuffle,
    )


def describe_batch(batch: dict) -> None:
    """Print keys + shapes of one batch — handy for inspecting a dataloader batch."""
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k:32s} {tuple(v.shape)}  {v.dtype}")
        elif isinstance(v, list):
            print(f"  {k:32s} list[{len(v)}]  e.g. {v[0]!r}")
        else:
            print(f"  {k:32s} {type(v).__name__}")
