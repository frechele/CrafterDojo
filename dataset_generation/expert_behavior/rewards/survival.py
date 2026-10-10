from functools import partial
from typing import Union

import chex
import jax

from craftax.craftax_classic.constants import BlockType

from dataset_generation.expert_behavior.rewards.base import RewardWrapper
from dataset_generation.expert_behavior.rewards.registry import register_reward


@register_reward("survival")
class SurvivalRewardWrapper(RewardWrapper):
    """Base reward with a death penalty when the player dies or steps on lava."""

    def __init__(self, env, death_penalty: int = 0):
        super().__init__(env)
        self.death_penalty = death_penalty

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, key: chex.PRNGKey, params=None):
        return self._env.reset(key, params)

    @partial(jax.jit, static_argnums=(0, 4))
    def step(self, key: chex.PRNGKey, state, action: Union[int, float], params=None):
        obs, env_state, reward, done, info = self._env.step(key, state, action, params)

        is_lava = (
            env_state.map[env_state.player_position[0], env_state.player_position[1]]
            == BlockType.LAVA.value
        )
        is_dead = env_state.player_health <= 0

        reward = reward - self.death_penalty * (is_lava | is_dead)
        return obs, env_state, reward, done, info
