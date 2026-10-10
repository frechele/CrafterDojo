"""Optimizer param-group helpers for MineCLIP (layerwise LR decay)."""

import torch
from omegaconf import DictConfig

from .mineclip import MineCLIP


VISION_BLOCKS_PREFIX = "image_encoder.blocks"
TEXT_BLOCKS_PREFIX = "clip_model.text_model.blocks"


def _calculate_block_lr(name, blocks, base_lr, layerwise_lr_decay, pretrained_layers_lr_multiplier):
    try:
        tokens = name.split(".")
        block_idx = int(tokens[tokens.index("blocks") + 1])
    except Exception:
        block_idx = None

    total_blocks = len(blocks)
    if (block_idx is not None) and (block_idx < total_blocks):
        exponent = 0 if block_idx == (total_blocks - 1) else 1
        return base_lr * pretrained_layers_lr_multiplier * (layerwise_lr_decay**exponent)
    return base_lr


def get_param_groups(model: MineCLIP, config: DictConfig) -> list[dict]:
    """Layer-wise LR: pretrained vision/text blocks get reduced, last block is full-multiplied."""
    base_lr = config.learning_rate
    layerwise_lr_decay = config.layerwise_lr_decay
    pretrained_layers_lr_multiplier = config.pretrained_layers_lr_multiplier

    param_groups: dict[float, list[torch.nn.Parameter]] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if VISION_BLOCKS_PREFIX in name:
            block_lr = _calculate_block_lr(
                name,
                model.image_encoder.blocks,
                base_lr,
                layerwise_lr_decay,
                pretrained_layers_lr_multiplier,
            )
        elif TEXT_BLOCKS_PREFIX in name:
            block_lr = _calculate_block_lr(
                name,
                model.clip_model.text_model.blocks,
                base_lr,
                layerwise_lr_decay,
                pretrained_layers_lr_multiplier,
            )
        else:
            block_lr = base_lr

        param_groups.setdefault(block_lr, []).append(param)

    return [{"params": params, "lr": lr} for lr, params in param_groups.items()]
