from functools import partial
from typing import Union

import chex
import jax

from dataset_generation.expert_behavior.rewards.base import RewardWrapper
from dataset_generation.expert_behavior.rewards.registry import register_reward


@register_reward("crafter")
class CrafterRewardWrapper(RewardWrapper):
    """Pass-through: uses the base Craftax achievement + health reward as-is."""

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, key: chex.PRNGKey, params=None):
        return self._env.reset(key, params)

    @partial(jax.jit, static_argnums=(0, 4))
    def step(self, key: chex.PRNGKey, state, action: Union[int, float], params=None):
        return self._env.step(key, state, action, params)
