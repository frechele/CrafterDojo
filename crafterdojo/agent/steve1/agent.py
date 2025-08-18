import torch

from crafterdojo.lib.VPT.agent import load_model_parameters
from crafterdojo.lib.steve1.config import PRIOR_INFO
from crafterdojo.lib.steve1.data.text_alignment.vae import load_vae_model
from crafterdojo.lib.steve1.utils.embed_utils import get_prior_embed
from crafterdojo.lib.steve1.MineRLConditionalAgent import CrafterConditionalAgent


class Steve1Agent:
    def __init__(
        self,
        model: str,
        model_weights: str,
        prior_weights: str,
        cond_scale: float,
        mineclip,
        device: torch.device,
    ):
        self.mineclip = mineclip
        PRIOR_INFO["model_path"] = prior_weights
        self.prior = load_vae_model(PRIOR_INFO, device)
        self.device = device

        policy_kwargs, pi_head_kwargs, lora_kwargs = load_model_parameters(model)

        self.agent = CrafterConditionalAgent(device, policy_kwargs, pi_head_kwargs, lora_kwargs)

        self.agent.load_weights(model_weights)
        self.agent.policy.eval()
        self.agent.policy = torch.compile(self.agent.policy)

        self.cond_scale = cond_scale
        self.goal_embed = None
    
    def reset(self):
        self.agent.reset(self.cond_scale)

    def set_goal(self, goal: torch.Tensor | str):
        if isinstance(goal, str):
            self.goal_embed = get_prior_embed(goal, self.mineclip, self.prior, self.device)
        else:
            self.goal_embed = goal.reshape((1, -1))
    
    def get_action(self, obs: torch.Tensor, greedy: bool = False):
        action = self.agent.get_action({"pov": obs["img"]}, self.goal_embed, greedy=greedy)
        return action
