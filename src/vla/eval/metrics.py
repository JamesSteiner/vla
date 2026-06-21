"""Metrics shared across all three heads. Success rate comes from the rollout loop
(env-dependent); these are the head-agnostic pieces.

Reported per head: task success rate (%), inference latency (ms/chunk), action
smoothness (mean ||a_{t+1} - a_t||), validation L1.
"""

from __future__ import annotations

import time
from contextlib import contextmanager

import torch
from torch import Tensor


@contextmanager
def Timer(store: list[float]):
    """Append elapsed milliseconds to `store`. Wrap a single predict_chunk call.

    Use this around the head's inference so latency is measured the same way for
    all three (regression = 1 pass; flow = num_steps; diffusion = DDIM steps).
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        store.append((time.perf_counter() - t0) * 1000.0)


def action_smoothness(actions: Tensor) -> float:
    """mean ||a_{t+1} - a_t|| over a rollout. actions: [T, action_dim]."""
    if actions.shape[0] < 2:
        return 0.0
    return (actions[1:] - actions[:-1]).norm(dim=-1).mean().item()


def chunk_l1(pred_chunk: Tensor, target_chunk: Tensor) -> float:
    """Validation L1 over an action chunk."""
    return (pred_chunk - target_chunk).abs().mean().item()
