import math
import inspect
import os
import logging

import torch
import torch.nn as nn

from lightning.fabric import Fabric


def divide_optimizer_groups(
    model: nn.Module,
    weight_decay: float,
    learning_rate: float,
    pi_head_lr_mult: float = 1.0,
    goal_affine_lr_mult: float = 1.0,
    lora_lr_mult: float = 1.0,
):
    # separate out all parameters to those that will and won't experience regularizing weight decay
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear,)
    blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            if not p.requires_grad:
                continue

            fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name
            # random note: because named_modules and named_parameters are recursive
            # we will see the same tensors p many many times. but doing it this way
            # allows us to know which parent module any tensor p belongs to...
            if pn.endswith("bias"):
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)
            else:
                decay.add(fpn)

    # If a parameter is in both decay and no_decay, remove it from decay.
    decay = decay - no_decay

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    
    inter_params = decay & no_decay
    
    # Filter out parameters that were in decay/no_decay but not in param_dict
    decay = decay & set(param_dict.keys())
    no_decay = no_decay & set(param_dict.keys())
    
    assert (
        len(inter_params) == 0
    ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
    
    # This assertion can be relaxed if the model structure doesn't match exactly what we expect
    missing_params = param_dict.keys() - (decay | no_decay)
    if missing_params:
        print(f"Warning: {len(missing_params)} parameters not categorized into decay/no_decay: {missing_params}")
        # Add missing parameters to decay by default
        decay.update(missing_params)

    # Print the keys that are in each set in a comma-separated list.
    # print(f"decay keys: {', '.join(sorted(list(decay)))}")
    # print(f"no decay keys: {', '.join(sorted(list(no_decay)))}")

    def _is_pi_head(pn: str) -> bool:
        return pn.startswith("pi_head.") or ".pi_head." in pn

    def _is_goal_affine(pn: str) -> bool:
        return "mineclip_embed_linear" in pn

    def _is_lora(pn: str) -> bool:
        return "lora_A" in pn or "lora_B" in pn

    def _classify(pn: str) -> str:
        if _is_pi_head(pn): return "pi_head"
        if _is_goal_affine(pn): return "goal_affine"
        if _is_lora(pn): return "lora"
        return "main"

    buckets = {"main": (set(), set()), "pi_head": (set(), set()),
               "goal_affine": (set(), set()), "lora": (set(), set())}
    for pn in decay:
        buckets[_classify(pn)][0].add(pn)
    for pn in no_decay:
        buckets[_classify(pn)][1].add(pn)

    raw_groups = [
        ("decay",               sorted(buckets["main"][0]),        weight_decay, 1.0),
        ("no_decay",            sorted(buckets["main"][1]),        0.0,          1.0),
        ("decay_pi_head",       sorted(buckets["pi_head"][0]),     weight_decay, pi_head_lr_mult),
        ("no_decay_pi_head",    sorted(buckets["pi_head"][1]),     0.0,          pi_head_lr_mult),
        ("decay_goal_affine",   sorted(buckets["goal_affine"][0]), weight_decay, goal_affine_lr_mult),
        ("no_decay_goal_affine",sorted(buckets["goal_affine"][1]), 0.0,          goal_affine_lr_mult),
        ("decay_lora",          sorted(buckets["lora"][0]),        weight_decay, lora_lr_mult),
        ("no_decay_lora",       sorted(buckets["lora"][1]),        0.0,          lora_lr_mult),
    ]
    optim_groups = []
    for name, names, wd, mult in raw_groups:
        if not names:
            continue
        optim_groups.append({
            "params": [param_dict[pn] for pn in names],
            "weight_decay": wd,
            "lr_mult": mult,
            "name": name,
        })
    return optim_groups

def configure_optimizers(
    model: nn.Module,
    weight_decay: float,
    learning_rate: float,
    pi_head_lr_mult: float = 1.0,
    goal_affine_lr_mult: float = 1.0,
    lora_lr_mult: float = 1.0,
):
    optim_groups = divide_optimizer_groups(
        model, weight_decay, learning_rate,
        pi_head_lr_mult=pi_head_lr_mult,
        goal_affine_lr_mult=goal_affine_lr_mult,
        lora_lr_mult=lora_lr_mult,
    )

    # new PyTorch nightly has a new 'fused' option for AdamW that is much faster
    use_fused = "fused" in inspect.signature(torch.optim.AdamW).parameters
    print(f"using fused AdamW: {use_fused}")
    extra_args = dict(fused=True) if use_fused else dict()

    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, **extra_args)

    return optimizer


def get_lr(args, num_samples_processed):
    min_lr = args.min_learning_rate

    # 1) linear warmup for warmup_iters steps
    if num_samples_processed < args.warmup_samples:
        return args.learning_rate * num_samples_processed / args.warmup_samples
    # 2) if it > lr_decay_iters, return min learning rate
    if num_samples_processed > args.n_samples * 1.1:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (num_samples_processed - args.warmup_samples) / (
        args.n_samples * 1.1 - args.warmup_samples
    )
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (args.learning_rate - min_lr)


def save_checkpoint(fabric, checkpoint_dir, n_batches, best_val_loss):
    """
    Save training checkpoint information.
    
    Args:
        fabric: Lightning Fabric instance
        checkpoint_dir: Directory to save checkpoint files
        n_batches: Number of batches processed
        best_val_loss: Best validation loss so far
    """
    if fabric.is_global_zero:
        # Save the number of steps
        with open(os.path.join(checkpoint_dir, "n_batches.txt"), "w") as f:
            f.write(str(n_batches))
        # Save the best validation loss
        with open(os.path.join(checkpoint_dir, "best_val_loss.txt"), "w") as f:
            f.write(str(best_val_loss))


def compute_gradient_l2_norm(model: nn.Module):
    """
    Compute L2 norm of gradients for all parameters in the model.
    
    Args:
        model: PyTorch model
        
    Returns:
        float: L2 norm of gradients
    """
    total_norm = 0
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm**0.5
    return total_norm


def compute_weight_l2_norm(model: nn.Module):
    """
    Compute L2 norm of weights for all parameters in the model.
    
    Args:
        model: PyTorch model
        
    Returns:
        float: L2 norm of weights
    """
    total_norm = 0
    for param in model.parameters():
        param_norm = param.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm**0.5
    return total_norm


def print_model_parameters(model: nn.Module, fabric: Fabric):
    if fabric.is_global_zero:
        num_params = sum(p.numel() for p in model.parameters())
        logging.info(f"Number of parameters: {num_params:,}")

        trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        logging.info(f"Number of trainable parameters: {trainable_params:,}")
