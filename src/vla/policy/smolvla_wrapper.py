"""SmolVLA's normalization processors for a given dataset.

The controlled comparison loads pretrained SmolVLA with LeRobot's `make_policy` (which
adapts the action/state/image features to the dataset) and trains only the action expert.
The one helper needed on top of that is the preprocessor/postprocessor pair below: it
renames the raw dataloader batch to the policy's feature keys, tokenises the language
`task`, and normalises state/action. The policy's `forward` and the regression/diffusion
samplers all expect a batch in that form.
"""

from __future__ import annotations

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


def make_smolvla_processors(policy: SmolVLAPolicy, dataset_stats: dict):
    """Build SmolVLA's (preprocessor, postprocessor) for a dataset's stats.

    Run `batch = preprocessor(batch)` before policy.forward / the head samplers, and
    `actions = postprocessor(raw_actions)` to unnormalise sampled actions.
    `dataset_stats` comes from `dataset.meta.stats`.
    """
    from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors

    return make_smolvla_pre_post_processors(policy.config, dataset_stats=dataset_stats)
