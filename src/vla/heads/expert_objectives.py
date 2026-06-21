"""Objectives that reuse SmolVLA's action EXPERT, varying ONLY the objective.

Each objective feeds action tokens through the SAME expert (`vlm_with_expert`) and decodes
with `action_out_proj`, so the architecture (transformer + per-layer cross-attention to the
prefix) is held fixed. Only the TRAINING LOSS and the INFERENCE procedure differ:

  flow (native)  : noisy action x_t = t*noise+(1-t)*a ; predict velocity ; MSE  (policy.forward)
  regression     : deterministic query (zeros, t=0)   ; predict action   ; L1   (here)
  diffusion      : DDPM-noised action                 ; predict eps      ; MSE

All three train the same params (the ~100M expert + action projections), so a success gap is
attributable to the OBJECTIVE, not the architecture. (An external pooled-MLP head on the prefix
would confound objective with architecture; the expert is deliberately reused to avoid that.)

Mirrors VLAFlowMatching.forward (modeling_smolvla.py); assumes `batch` already went through the
SmolVLA preprocessor (normalized + tokenized + on device), same as policy.forward.
"""

from __future__ import annotations

import torch

from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS


def _expert_decode(policy, batch, x_t: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    """Run prefix + suffix(x_t, time) through the expert; decode -> [B, chunk, max_action_dim]."""
    model = policy.model
    images, img_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    lang_tokens = batch[OBS_LANGUAGE_TOKENS]
    lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

    prefix_embs, prefix_pad, prefix_att = model.embed_prefix(images, img_masks, lang_tokens, lang_masks, state=state)
    suffix_embs, suffix_pad, suffix_att = model.embed_suffix(x_t, time)
    pad_masks = torch.cat([prefix_pad, suffix_pad], dim=1)
    att_masks = torch.cat([prefix_att, suffix_att], dim=1)
    att_2d = make_att_2d_masks(pad_masks, att_masks)
    position_ids = torch.cumsum(pad_masks, dim=1) - 1
    (_, suffix_out), _ = model.vlm_with_expert.forward(
        attention_mask=att_2d,
        position_ids=position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, suffix_embs],
        use_cache=False,
        fill_kv_cache=False,
    )
    suffix_out = suffix_out[:, -model.config.chunk_size :].to(torch.float32)
    return model.action_out_proj(suffix_out)  # [B, chunk, max_action_dim]


def _masked_l1(pred: torch.Tensor, target: torch.Tensor, action_is_pad: torch.Tensor | None) -> torch.Tensor:
    l1 = (pred - target).abs()  # [B, chunk, adim]
    if action_is_pad is None:
        return l1.mean()
    valid = (~action_is_pad).unsqueeze(-1).to(l1.dtype)  # [B, chunk, 1]; 0 on padded steps
    return (l1 * valid).sum() / (valid.sum().clamp_min(1.0) * pred.shape[-1])


def regression_loss(policy, batch) -> torch.Tensor:
    """Deterministic forward (zero query, t=0) -> action; masked L1 against the target chunk."""
    actions = policy.prepare_action(batch)  # [B, chunk, max_action_dim] normalized + padded
    x_t = torch.zeros_like(actions)
    time = torch.zeros(actions.shape[0], device=actions.device)
    pred = _expert_decode(policy, batch, x_t, time)
    adim = policy.config.action_feature.shape[0]
    return _masked_l1(pred[:, :, :adim], actions[:, :, :adim], batch.get("action_is_pad"))


@torch.no_grad()
def regression_sample(policy, batch) -> torch.Tensor:
    """Inference: ONE deterministic forward -> normalized action chunk [B, chunk, action_dim]."""
    model = policy.model
    state = policy.prepare_state(batch)
    bsize, device = state.shape[0], state.device
    x_t = torch.zeros(bsize, model.config.chunk_size, model.config.max_action_dim, device=device)
    time = torch.zeros(bsize, device=device)
    pred = _expert_decode(policy, batch, x_t, time)
    return pred[:, :, : policy.config.action_feature.shape[0]]  # normalized; caller unnormalizes


# --- diffusion (DDPM train / DDIM infer), same expert; predicts noise epsilon ----------------

_SCHED_CACHE: dict = {}


def _scheduler(kind: str, diff_cfg: dict):
    """Cached DDPM (train) / DDIM (infer) scheduler from the diffusion config."""
    T = int(diff_cfg.get("train_timesteps", 100))
    beta = diff_cfg.get("beta_schedule", "squaredcos_cap_v2")
    steps = int(diff_cfg.get("infer_steps", 10))
    key = (kind, T, beta, steps)
    if key not in _SCHED_CACHE:
        from diffusers import DDIMScheduler, DDPMScheduler

        cls = DDPMScheduler if kind == "ddpm" else DDIMScheduler
        sched = cls(num_train_timesteps=T, beta_schedule=beta, prediction_type="epsilon", clip_sample=False)
        if kind == "ddim":
            sched.set_timesteps(steps)
        _SCHED_CACHE[key] = sched
    return _SCHED_CACHE[key]


def _masked_mse(pred: torch.Tensor, target: torch.Tensor, action_is_pad: torch.Tensor | None) -> torch.Tensor:
    se = (pred - target) ** 2
    if action_is_pad is None:
        return se.mean()
    valid = (~action_is_pad).unsqueeze(-1).to(se.dtype)
    return (se * valid).sum() / (valid.sum().clamp_min(1.0) * pred.shape[-1])


def diffusion_loss(policy, batch, diff_cfg: dict | None = None) -> torch.Tensor:
    """DDPM: noise the action at a random timestep; the expert predicts epsilon; masked MSE."""
    sched = _scheduler("ddpm", diff_cfg or {})
    actions = policy.prepare_action(batch)  # [B, chunk, max_action_dim] normalized + padded
    noise = torch.randn_like(actions)
    t = torch.randint(0, sched.config.num_train_timesteps, (actions.shape[0],), device=actions.device)
    x_t = sched.add_noise(actions, noise, t)
    time = t.float() / sched.config.num_train_timesteps  # -> [0,1] for embed_suffix (same as flow)
    eps_pred = _expert_decode(policy, batch, x_t, time)
    adim = policy.config.action_feature.shape[0]
    return _masked_mse(eps_pred[:, :, :adim], noise[:, :, :adim], batch.get("action_is_pad"))


@torch.no_grad()
def diffusion_sample(policy, batch, diff_cfg: dict | None = None) -> torch.Tensor:
    """DDIM: compute the prefix KV ONCE, then `infer_steps` denoise steps. -> [B, chunk, action_dim]."""
    diff_cfg = diff_cfg or {}
    ddim = _scheduler("ddim", diff_cfg)
    model = policy.model

    images, img_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    lang_tokens = batch[OBS_LANGUAGE_TOKENS]
    lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
    prefix_embs, prefix_pad, prefix_att = model.embed_prefix(images, img_masks, lang_tokens, lang_masks, state=state)
    prefix_2d = make_att_2d_masks(prefix_pad, prefix_att)
    prefix_pos = torch.cumsum(prefix_pad, dim=1) - 1
    _, past_kv = model.vlm_with_expert.forward(
        attention_mask=prefix_2d,
        position_ids=prefix_pos,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=model.config.use_cache,
        fill_kv_cache=True,
    )

    bsize, device = state.shape[0], state.device
    T = int(diff_cfg.get("train_timesteps", 100))  # normalize DDIM's discrete t -> [0,1] for embed_suffix
    x_t = torch.randn(bsize, model.config.chunk_size, model.config.max_action_dim, device=device)
    for t in ddim.timesteps:
        time = torch.full((bsize,), float(t) / T, device=device)
        eps = model.denoise_step(prefix_pad, past_kv, x_t, time)  # reuses cached prefix -> epsilon
        x_t = ddim.step(eps, t, x_t).prev_sample
    return x_t[:, :, : policy.config.action_feature.shape[0]]  # normalized; caller unnormalizes
