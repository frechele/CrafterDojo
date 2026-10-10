"""High-level Steve-1 agents: wrap CrafterConditionalAgent with goal encoders."""

import torch

from crafterdojo.agent.steve1.conditional_agent import CrafterConditionalAgent
from crafterdojo.model.steve1.config import PRIOR_INFO
from crafterdojo.model.steve1.data.text_alignment.vae import load_vae_model
from crafterdojo.model.steve1.utils.embed_utils import get_prior_embed, get_sbert_embed
from crafterdojo.model.vpt.config import load_model_parameters


class Steve1Agent:
    def __init__(
        self,
        model: str,
        model_weights: str,
        prior_weights: str,
        cond_scale: float,
        mineclip,
        device: torch.device,
        cond_position: str = "output",
    ):
        self.mineclip = mineclip
        PRIOR_INFO["model_path"] = prior_weights
        self.prior = load_vae_model(PRIOR_INFO, device)
        self.device = device

        policy_kwargs, pi_head_kwargs, lora_kwargs = load_model_parameters(model)
        policy_kwargs["cond_position"] = cond_position

        self.agent = CrafterConditionalAgent(
            device, policy_kwargs, pi_head_kwargs, lora_kwargs
        )

        self.agent.load_weights(model_weights)
        self.agent.policy.eval()
        self.agent.policy = torch.compile(self.agent.policy)

        self.cond_scale = cond_scale
        self.goal_embed = None

    def reset(self):
        self.agent.reset(self.cond_scale)

    def set_goal(self, goal: torch.Tensor | str):
        if isinstance(goal, str):
            self.goal_embed = get_prior_embed(
                goal, self.mineclip, self.prior, self.device
            )
        else:
            self.goal_embed = goal.reshape((1, -1))

    def get_action(
        self,
        obs: torch.Tensor,
        greedy: bool = False,
        mask_place: bool = False,
        mask_craft: bool = False,
    ):
        return self.agent.get_action(
            {"pov": obs["img"]},
            self.goal_embed,
            greedy=greedy,
            mask_place=mask_place,
            mask_craft=mask_craft,
        )


class SBERTSteve1Agent:
    """Steve-1 agent using SentenceBERT for goal conditioning (no MineCLIP/VAE prior)."""

    def __init__(
        self,
        model: str,
        model_weights: str,
        cond_scale: float,
        sbert_model_name: str,
        device: torch.device,
        goal_embed_dim: int = 768,
        cond_position: str = "output",
    ):
        from sentence_transformers import SentenceTransformer

        self.sbert = SentenceTransformer(sbert_model_name)
        self.device = device

        policy_kwargs, pi_head_kwargs, lora_kwargs = load_model_parameters(model)
        policy_kwargs["goal_embed_dim"] = goal_embed_dim
        policy_kwargs["cond_position"] = cond_position

        self.agent = CrafterConditionalAgent(
            device, policy_kwargs, pi_head_kwargs, lora_kwargs
        )

        self.agent.load_weights(model_weights)
        self.agent.policy.eval()
        self.agent.policy = torch.compile(self.agent.policy)

        self.cond_scale = cond_scale
        self.goal_embed = None

    def reset(self):
        self.agent.reset(self.cond_scale)

    def set_goal(self, goal: torch.Tensor | str):
        if isinstance(goal, str):
            self.goal_embed = get_sbert_embed(goal, self.sbert)
        else:
            self.goal_embed = goal.reshape((1, -1))

    def get_action(
        self,
        obs: torch.Tensor,
        greedy: bool = False,
        mask_place: bool = False,
        mask_craft: bool = False,
    ):
        return self.agent.get_action(
            {"pov": obs["img"]},
            self.goal_embed,
            greedy=greedy,
            mask_place=mask_place,
            mask_craft=mask_craft,
        )
