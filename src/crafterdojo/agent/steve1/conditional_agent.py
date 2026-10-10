"""Goal-conditioned VPT agent for Steve-1 (classifier-free guidance at act-time)."""

import numpy as np
import torch as th
from gym3.types import discrete_scalar

from crafterdojo.agent.vpt.crafter_rl_agent import CrafterRLAgent
from crafterdojo.model.steve1.embed_conditioned_policy import MinecraftAgentPolicy
from crafterdojo.model.vpt.config import PI_HEAD_KWARGS, POLICY_KWARGS
from crafterdojo.model.vpt.lib.torch_util import (
    default_device_type,
    set_default_torch_device,
)


ACTION_NAMES = (
    "noop",
    "move_left",
    "move_right",
    "move_up",
    "move_down",
    "do",
    "sleep",
    "place_stone",
    "place_table",
    "place_furnace",
    "place_plant",
    "make_wood_pickaxe",
    "make_stone_pickaxe",
    "make_iron_pickaxe",
    "make_wood_sword",
    "make_stone_sword",
    "make_iron_sword",
)
PLACE_ACTION_INDICES = tuple(
    idx for idx, name in enumerate(ACTION_NAMES) if name.startswith("place_")
)
CRAFT_ACTION_INDICES = tuple(
    idx for idx, name in enumerate(ACTION_NAMES) if name.startswith("make_")
)


class CrafterConditionalAgent(CrafterRLAgent):
    def __init__(
        self,
        device=None,
        policy_kwargs=None,
        pi_head_kwargs=None,
        lora_kwargs=None,
    ):
        if device is None:
            device = default_device_type()
        self.device = th.device(device)
        set_default_torch_device(self.device)
        action_space = discrete_scalar(17)

        if policy_kwargs is None:
            policy_kwargs = POLICY_KWARGS
        if pi_head_kwargs is None:
            pi_head_kwargs = PI_HEAD_KWARGS

        agent_kwargs = dict(
            policy_kwargs=policy_kwargs,
            pi_head_kwargs=pi_head_kwargs,
            action_space=action_space,
        )

        self.policy = MinecraftAgentPolicy(**agent_kwargs).to(device)

        if lora_kwargs is not None:
            self.policy.inject_lora(lora_kwargs["alpha"], lora_kwargs["r"])

        self.num_actions = self.policy.pi_head.num_actions
        if self.num_actions != len(ACTION_NAMES):
            raise ValueError(
                f"Action space size ({self.num_actions}) does not match ACTION_NAMES ({len(ACTION_NAMES)})."
            )

        self.hidden_state = self.policy.initial_state(1)
        self._dummy_first = th.from_numpy(np.array((False,))).to(device)

    def reset(self, cond_scale=None):
        if cond_scale is None or cond_scale == 0:
            self.hidden_state = self.policy.initial_state(1)
        else:
            self.hidden_state = self.policy.initial_state(2)
        self.cond_scale = cond_scale

    def get_agent_input_pov(self, frame: np.ndarray):
        return frame[None]

    def take_action_on_frame(self, agent_input_pov: np.ndarray):
        agent_input = {"img": th.from_numpy(agent_input_pov).to(self.device)}
        agent_action, self.hidden_state, _ = self.policy.act(
            agent_input, self._dummy_first, self.hidden_state, stochastic=True
        )
        return agent_action

    def get_action(
        self,
        minerl_obs,
        goal_embed,
        greedy: bool = False,
        mask_place: bool = False,
        mask_craft: bool = False,
    ):
        action_mask = self._build_action_mask(
            mask_place=mask_place, mask_craft=mask_craft
        )
        agent_input = self._env_obs_to_agent(minerl_obs, goal_embed, mask=action_mask)
        agent_action, self.hidden_state, _ = self.policy.act(
            agent_input,
            self._dummy_first,
            self.hidden_state,
            stochastic=not greedy,
            cond_scale=self.cond_scale,
        )
        return agent_action

    def _env_obs_to_agent(self, minerl_obs, goal_embed, device=None, mask=None):
        if device is None:
            device = self.device

        agent_input = {"img": th.from_numpy(minerl_obs["pov"][None]).to(device)}
        agent_input["goal_embed"] = th.from_numpy(goal_embed).to(device)
        if mask is not None:
            agent_input["mask"] = mask.to(device)
        return agent_input

    def _build_action_mask(self, mask_place: bool, mask_craft: bool):
        if not (mask_place or mask_craft):
            return None

        mask = th.ones((1, self.num_actions), dtype=th.bool, device=self.device)

        if mask_place and PLACE_ACTION_INDICES:
            mask[:, PLACE_ACTION_INDICES] = False

        if mask_craft and CRAFT_ACTION_INDICES:
            mask[:, CRAFT_ACTION_INDICES] = False

        return mask
