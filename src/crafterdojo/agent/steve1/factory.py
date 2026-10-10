"""Shared Steve-1 conditional-agent construction for BC training."""

from crafterdojo.agent.steve1.conditional_agent import CrafterConditionalAgent
from crafterdojo.model.vpt.config import load_model_parameters


def build_conditional_agent(config, device, *, freeze_value_head: bool):
    """Load kwargs, inject goal_embed_dim, build CrafterConditionalAgent."""
    policy_kwargs, pi_head_kwargs, lora_kwargs = load_model_parameters(config.model)

    goal_embed_dim = getattr(config, "goal_embed_dim", 512)
    policy_kwargs["goal_embed_dim"] = goal_embed_dim
    policy_kwargs["cond_position"] = getattr(config, "cond_position", "output")

    agent = CrafterConditionalAgent(device, policy_kwargs, pi_head_kwargs, lora_kwargs)

    if config.weights is not None:
        agent.load_weights(config.weights)
    else:
        agent.policy.reset_parameters()

    if freeze_value_head:
        for _, param in agent.policy.value_head.named_parameters():
            param.requires_grad = False

    return agent
