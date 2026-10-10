"""MineCLIP construction for CraftCLIP training (weight loading + freeze policy)."""

import os
import torch.nn as nn

from .mineclip import MineCLIP


def _set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for param in module.parameters():
        param.requires_grad = requires_grad


def _apply_partial_freeze(model: MineCLIP, num_tunable_blocks: int = 2) -> None:
    """Freeze vision/text encoders except the last `num_tunable_blocks` blocks."""
    _set_requires_grad(model.clip_model.vision_model, False)
    for block in model.clip_model.vision_model.blocks[-num_tunable_blocks:]:
        _set_requires_grad(block, True)

    _set_requires_grad(model.clip_model.text_model, False)
    for block in model.clip_model.text_model.blocks[-num_tunable_blocks:]:
        _set_requires_grad(block, True)


def build_mineclip_for_training(config) -> MineCLIP:
    """Load OpenAI CLIP weights into a MineCLIP and freeze all but the last 2 blocks."""
    model = MineCLIP(**config.mineclip)
    if config.oai_clip_weights is not None:
        weights_path = str(config.oai_clip_weights)
        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"OpenAI CLIP weights not found: {weights_path}. "
                "CrafterCLIP training should not silently start from a random CLIP backbone."
            )
        # Accepts the original OpenAI release (``ViT-B-16.pt``, converted on the fly)
        # or a pre-converted state dict.
        from crafterdojo.evaluator.crafterclip.loaders import load_openai_clip_state_dict

        clip_state_dict = load_openai_clip_state_dict(weights_path, list(config.mineclip.resolution))
        model.clip_model.load_state_dict(clip_state_dict)
    _apply_partial_freeze(model, num_tunable_blocks=2)
    return model
