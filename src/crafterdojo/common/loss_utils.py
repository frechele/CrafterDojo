"""Shared loss / chunking helpers for BC training runners."""

import torch
from torch.distributions import Categorical


def get_chunk(x, t: int, trunc_t: int):
    if isinstance(x, torch.Tensor):
        return x[:, t : t + trunc_t]
    return [y[t : t + trunc_t] for y in x]


def compute_entropy(pi_distribution: torch.Tensor) -> torch.Tensor:
    return Categorical(logits=pi_distribution).entropy().mean()


def masked_sum(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(device=values.device, dtype=values.dtype)
    return (values * mask).sum()


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.to(device=values.device, dtype=values.dtype).sum().clamp_min(1.0)
    return masked_sum(values, mask) / denom


def categorical_accuracy_sum(
    pi_distribution: torch.Tensor,
    actions: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Unreduced count of argmax-matches over valid positions."""
    match = (pi_distribution.argmax(dim=-1) == actions).to(dtype=torch.float32)
    mask = valid_mask.to(device=match.device, dtype=torch.float32)
    return (match * mask).sum()


def categorical_entropy_sum(
    pi_distribution: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Sum of per-step categorical entropy over valid positions."""
    entropy = Categorical(logits=pi_distribution).entropy()
    return masked_sum(entropy, valid_mask)
