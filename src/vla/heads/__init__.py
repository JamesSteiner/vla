"""Action heads for the controlled comparison.

The three conditions — flow / regression / diffusion — are implemented in
``expert_objectives.py`` as alternative TRAINING OBJECTIVES + INFERENCE procedures on the
SHARED SmolVLA action expert. Only the objective (and sampling) varies; the architecture
(transformer + per-layer cross-attention to the prefix + ``action_out_proj``) is held fixed,
so a success gap is attributable to the objective, not the architecture.

  flow       : native flow-matching MSE              (lerobot SmolVLAPolicy.forward)
  regression : deterministic forward (zeros, t=0)+L1  (expert_objectives.regression_*)
  diffusion  : DDPM train / DDIM infer (predict eps)  (expert_objectives.diffusion_*)
"""
