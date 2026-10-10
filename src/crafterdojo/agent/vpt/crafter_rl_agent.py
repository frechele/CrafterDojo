"""Env-facing VPT agent: hidden-state tracking, observation prep, action sampling."""

import numpy as np
import torch as th
from gym3.types import discrete_scalar

from crafterdojo.common.checkpoint import load_state_dict
from crafterdojo.model.vpt.config import PI_HEAD_KWARGS, POLICY_KWARGS
from crafterdojo.model.vpt.lib.policy import MinecraftAgentPolicy
from crafterdojo.model.vpt.lib.torch_util import (
    default_device_type,
    set_default_torch_device,
)


class CrafterRLAgent:
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
            self.policy.inject_lora(
                lora_alpha=lora_kwargs["alpha"], lora_r=lora_kwargs["r"]
            )

        self.hidden_state = self.policy.initial_state(1)
        self._dummy_first = th.from_numpy(np.array((False,))).to(device)

    def load_weights(self, path):
        self.policy.load_state_dict(
            load_state_dict(path, map_location=self.device),
            strict=False,
        )
        self.reset()

    def reset(self):
        self.hidden_state = self.policy.initial_state(1)

    def _env_obs_to_agent(self, minerl_obs):
        agent_input = minerl_obs["pov"][None]
        return {"img": th.from_numpy(agent_input).to(self.device)}

    def get_action(self, minerl_obs, return_result: bool = False):
        agent_input = self._env_obs_to_agent(minerl_obs)

        agent_action, self.hidden_state, result = self.policy.act(
            agent_input,
            self._dummy_first,
            self.hidden_state,
            stochastic=True,
            return_pd=True,
        )
        if return_result:
            return agent_action, result
        return agent_action
