"""Checkpoint loading helpers shared by all CrafterDojo models.

Released weights are distributed as ``.safetensors``. Checkpoints written by the
training runners (``*.weights`` / ``*.pt``) are plain PyTorch files and are still
supported so that locally trained models can be loaded the same way.
"""

from __future__ import annotations

import json
import os

import torch


def load_state_dict(path: str | os.PathLike, map_location="cpu") -> dict[str, torch.Tensor]:
    """Load a model state dict from ``.safetensors`` or a PyTorch checkpoint.

    Some modules share parameters (e.g. CrafterCLIP's image encoder is a view of
    its CLIP vision tower). ``safetensors`` stores each shared tensor once and
    records the remaining names in the ``aliases`` metadata entry; they are
    restored here so the result can be passed directly to ``load_state_dict``.
    """
    path = os.fspath(path)
    if path.endswith(".safetensors"):
        from safetensors import safe_open
        from safetensors.torch import load_file

        device = str(map_location) if map_location is not None else "cpu"
        state_dict = load_file(path, device=device)
        with safe_open(path, framework="pt") as f:
            metadata = f.metadata() or {}
        for alias, target in json.loads(metadata.get("aliases", "{}")).items():
            state_dict[alias] = state_dict[target]
        return state_dict

    state_dict = torch.load(path, map_location=map_location, weights_only=True)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    return state_dict


def load_model_config(path: str | os.PathLike) -> dict:
    """Load a VPT model-architecture file (``.yaml``; legacy pickled ``.model`` also accepted)."""
    path = os.fspath(path)
    if path.endswith((".yaml", ".yml")):
        import yaml

        with open(path) as f:
            return yaml.safe_load(f)

    import pickle

    with open(path, "rb") as f:
        return pickle.load(f)
