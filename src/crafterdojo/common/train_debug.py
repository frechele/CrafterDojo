"""Numerical-health diagnostics shared across BC training runners."""

import logging

import torch
from lightning.fabric import Fabric


def find_nonfinite_gradient_parameters(model, limit: int = 8) -> list[str]:
    names = []
    for name, param in model.named_parameters():
        grad = param.grad
        if grad is not None and not torch.isfinite(grad).all():
            names.append(name)
            if len(names) >= limit:
                break
    return names


def tensor_stats(tensor: torch.Tensor) -> dict:
    tensor = tensor.detach()
    stats = {
        "shape": tuple(tensor.shape),
        "dtype": str(tensor.dtype),
        "numel": int(tensor.numel()),
    }
    if tensor.numel() == 0:
        stats.update(
            {"nonfinite": 0, "min": "n/a", "max": "n/a", "mean": "n/a", "std": "n/a"}
        )
        return stats

    values = tensor.float()
    finite_mask = torch.isfinite(values)
    nonfinite = int((~finite_mask).sum().item())
    stats["nonfinite"] = nonfinite

    if finite_mask.any():
        finite_values = values[finite_mask]
        stats.update(
            {
                "min": float(finite_values.min().item()),
                "max": float(finite_values.max().item()),
                "mean": float(finite_values.mean().item()),
                "std": float(finite_values.std(unbiased=False).item()),
            }
        )
    else:
        stats.update(
            {
                "min": "all_nonfinite",
                "max": "all_nonfinite",
                "mean": "all_nonfinite",
                "std": "all_nonfinite",
            }
        )
    return stats


def format_tensor_stats(name: str, stats: dict) -> str:
    return (
        f"{name}: shape={stats['shape']} dtype={stats['dtype']} numel={stats['numel']} "
        f"nonfinite={stats['nonfinite']} min={stats['min']} max={stats['max']} "
        f"mean={stats['mean']} std={stats['std']}"
    )


def log_nonfinite_forward_debug(
    *,
    step: int,
    chunk_idx: int,
    obs_chunk,
    actions_chunk,
    firsts_chunk,
    pi_distribution=None,
    fabric: Fabric,
):
    if not fabric.is_global_zero:
        return

    logging.warning("Non-finite debug for step %s chunk %s", step, chunk_idx)
    logging.warning(format_tensor_stats("obs.img", tensor_stats(obs_chunk["img"])))
    logging.warning(format_tensor_stats("actions", tensor_stats(actions_chunk)))
    logging.warning(format_tensor_stats("firsts", tensor_stats(firsts_chunk)))
    if pi_distribution is not None:
        logging.warning(
            format_tensor_stats("pi_distribution", tensor_stats(pi_distribution))
        )


def summarize_gradient_parameters(model, limit: int = 8) -> list[str]:
    summaries = []
    for name, param in model.named_parameters():
        grad = param.grad
        if grad is None:
            continue
        stats = tensor_stats(grad)
        summaries.append((stats["nonfinite"], stats["max"], name, stats))

    def sort_key(item):
        nonfinite, max_value, _, _ = item
        finite_priority = 1 if nonfinite else 0
        if isinstance(max_value, str):
            max_value = float("-inf")
        return (finite_priority, max_value)

    summaries.sort(key=sort_key, reverse=True)
    return [
        format_tensor_stats(f"grad[{name}]", stats)
        for _, _, name, stats in summaries[:limit]
    ]


def maybe_clear_cuda_cache(device) -> None:
    if isinstance(device, torch.device) and device.type == "cuda":
        torch.cuda.empty_cache()
