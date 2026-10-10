"""Shared VPT agent construction for BC training and evaluation."""

import logging

from crafterdojo.agent.vpt.crafter_rl_agent import CrafterRLAgent
from crafterdojo.model.vpt.config import apply_model_overrides, load_model_parameters


def build_vpt_agent(config, device, *, freeze_value_head: bool, log: bool = False):
    """Load policy kwargs, construct CrafterRLAgent, optionally load weights.

    - freeze_value_head: disable grads on value_head (BC training does not need it).
    - log: emit an info line describing applied overrides.
    """
    policy_kwargs, pi_head_kwargs, lora_kwargs = load_model_parameters(config.model)

    overrides = getattr(config, "model_overrides", None)
    if overrides is not None:
        active = {k: v for k, v in overrides.items() if v is not None}
        if active:
            apply_model_overrides(policy_kwargs, overrides)
            if log:
                logging.info(f"Model overrides applied: {active}")

    pi_temp_override = getattr(config, "pi_head_temperature", None)
    if pi_temp_override is not None:
        pi_head_kwargs["temperature"] = float(pi_temp_override)
        if log:
            logging.info(f"pi_head temperature override: {pi_head_kwargs['temperature']}")

    agent = CrafterRLAgent(device, policy_kwargs, pi_head_kwargs, lora_kwargs)

    if config.weights is not None:
        agent.load_weights(config.weights)
    else:
        agent.policy.reset_parameters()

    if freeze_value_head:
        for _, param in agent.policy.value_head.named_parameters():
            param.requires_grad = False

    return agent
