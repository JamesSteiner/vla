"""Controlled action-head comparison on SmolVLA.

Three objectives on the SHARED SmolVLA action expert (`vla.heads.expert_objectives`):
  - flow matching  (native SmolVLA expert — the reference baseline)
  - regression     (deterministic L1)
  - diffusion      (DDPM train / DDIM inference)

See project-a-vla-action-head-comparison.md for the full spec.
"""

__all__ = ["heads", "policy", "data", "eval"]
