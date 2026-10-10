"""Classifier-free-guidance metrics for Steve-1: cond-vs-uncond divergence."""

import torch

from crafterdojo.common.loss_utils import masked_sum
from crafterdojo.model.vpt.lib.tree_util import tree_map


@torch.no_grad()
def compute_cfg_kl_sum(
    policy,
    obs_chunk: dict,
    agent_state,
    firsts_chunk: torch.Tensor,
    cond_pi_distribution: torch.Tensor,
    valid_mask: torch.Tensor,
    shape_bt: tuple[int, int],
) -> torch.Tensor:
    """Sum of per-step KL(uncond || cond) over valid positions.

    The unconditional branch zeros out `goal_embed` in the observation and reuses
    the provided `agent_state` / `firsts_chunk`. `cond_pi_distribution` must already
    be reshaped to (B, T, -1); `shape_bt` is the (B, T) used for that reshape.
    """
    uncond_obs_chunk = obs_chunk.copy()
    uncond_obs_chunk["goal_embed"] = torch.zeros_like(uncond_obs_chunk["goal_embed"])
    uncond_pi_distribution, _, _ = policy.get_output_for_observation(
        uncond_obs_chunk, agent_state, firsts_chunk
    )
    b, t = shape_bt
    uncond_pi_distribution = tree_map(
        lambda x: x.view(b, t, -1), uncond_pi_distribution
    )

    kl = policy.get_kl_of_action_dists(uncond_pi_distribution, cond_pi_distribution)
    # kl_divergence retains a trailing keepdim=True dim of size 1; align with valid_mask.
    if kl.dim() > valid_mask.dim():
        kl = kl.squeeze(-1)
    return masked_sum(kl, valid_mask)
