"""Weight-file helpers for CraftCLIP evaluators (OpenAI CLIP state-dict remap)."""

from collections import OrderedDict
from pathlib import Path

import torch

from crafterdojo.model.crafterclip.mineclip.pos_embed import interpolate_resize_pos_embed


OPENAI_CLIP_PATCH_SIZE = 16
DEFAULT_OPENAI_WEIGHT_CANDIDATES = (
    "models/clip/ViT-B-16.pt",
    "models/clip/ViT-B-16.weights",
)


def resolve_openai_weights_path(weights_path: str | None, *, repo_root: Path) -> str:
    if weights_path:
        return weights_path

    for candidate in DEFAULT_OPENAI_WEIGHT_CANDIDATES:
        candidate_path = repo_root / candidate
        if candidate_path.exists():
            return str(candidate_path)

    candidates = ", ".join(DEFAULT_OPENAI_WEIGHT_CANDIDATES)
    raise FileNotFoundError(f"OpenAI CLIP weights not found. Tried: {candidates}")


def convert_openai_state_dict(state_dict, target_resolution, patch_size: int = OPENAI_CLIP_PATCH_SIZE):
    """Remap an OpenAI CLIP state-dict to the internal `CLIP` model layout."""
    new_state_dict: OrderedDict = OrderedDict()

    new_state_dict["logit_scale"] = state_dict["logit_scale"]

    # vision model
    new_state_dict["vision_model.cls_token"] = state_dict["visual.class_embedding"]
    new_state_dict["vision_model.projection"] = state_dict["visual.proj"]
    new_state_dict["vision_model.conv1.weight"] = state_dict["visual.conv1.weight"]
    new_state_dict["vision_model.ln_pre.weight"] = state_dict["visual.ln_pre.weight"]
    new_state_dict["vision_model.ln_pre.bias"] = state_dict["visual.ln_pre.bias"]
    new_state_dict["vision_model.ln_post.weight"] = state_dict["visual.ln_post.weight"]
    new_state_dict["vision_model.ln_post.bias"] = state_dict["visual.ln_post.bias"]

    old_embed = state_dict["visual.positional_embedding"]
    cls_embed, old_embed = old_embed[:1], old_embed[1:]
    old_grid = int(old_embed.shape[0] ** 0.5)
    if old_grid * old_grid != old_embed.shape[0]:
        raise ValueError(f"Unexpected positional embedding shape: {old_embed.shape}")
    new_grid = (target_resolution[0] // patch_size, target_resolution[1] // patch_size)
    new_embed = interpolate_resize_pos_embed(old_embed, old_grid, new_grid)
    new_state_dict["vision_model.pos_embed"] = torch.cat([cls_embed, new_embed], dim=0)

    for orig_k, v in state_dict.items():
        if orig_k.startswith("visual.transformer.resblocks"):
            new_k = orig_k.replace("visual.transformer.resblocks", "vision_model.blocks")
            new_state_dict[new_k] = v

    # text model
    new_state_dict["text_model.token_embedding.weight"] = state_dict["token_embedding.weight"]
    new_state_dict["text_model.pos_embed"] = state_dict["positional_embedding"]
    new_state_dict["text_model.projection"] = state_dict["text_projection"]
    new_state_dict["text_model.ln_final.weight"] = state_dict["ln_final.weight"]
    new_state_dict["text_model.ln_final.bias"] = state_dict["ln_final.bias"]

    for orig_k, v in state_dict.items():
        if orig_k.startswith("transformer.resblocks"):
            new_k = orig_k.replace("transformer.resblocks", "text_model.blocks")
            new_state_dict[new_k] = v

    return new_state_dict


def load_openai_clip_state_dict(weights_path: str, target_resolution):
    if weights_path.endswith(".pt"):
        raw_state_dict = torch.jit.load(weights_path, map_location="cpu").state_dict()
        return convert_openai_state_dict(raw_state_dict, target_resolution)
    return torch.load(weights_path, map_location="cpu", weights_only=True)
