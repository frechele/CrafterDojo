from functools import partial
from typing import Any, Union

import chex
import jax
import jax.numpy as jnp
from flax import struct

from dataset_generation.expert_behavior.rewards.base import RewardWrapper
from dataset_generation.expert_behavior.rewards.registry import register_reward


@struct.dataclass
class CombatEnvState:
    env_state: Any
    prev_zombie_mask: jnp.ndarray
    prev_skeleton_mask: jnp.ndarray

    def __getattr__(self, name):
        return getattr(self.env_state, name)


# Craftax-Classic Action.DO (see craftax.craftax_classic.constants.Action).
_DO_ACTION = 5

# Facing offsets indexed by player_direction (NOOP, LEFT, RIGHT, UP, DOWN).
_FACING = jnp.array(
    [[0, 0], [0, -1], [0, 1], [-1, 0], [1, 0]], dtype=jnp.int32
)


@register_reward("combat")
class CombatRewardWrapper(RewardWrapper):
    """+alpha per zombie/skeleton killed via DO into the facing tile."""

    def __init__(self, env, alpha: float = 1.0):
        super().__init__(env)
        self.alpha = alpha

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, key: chex.PRNGKey, params=None):
        obs, env_state = self._env.reset(key, params)
        state = CombatEnvState(
            env_state=env_state,
            prev_zombie_mask=env_state.zombies.mask,
            prev_skeleton_mask=env_state.skeletons.mask,
        )
        return obs, state

    @partial(jax.jit, static_argnums=(0, 4))
    def step(self, key: chex.PRNGKey, state, action: Union[int, float], params=None):
        prev = state.env_state
        obs, env_state, base_reward, done, info = self._env.step(
            key, prev, action, params
        )

        facing_tile = prev.player_position + _FACING[prev.player_direction]  # (2,)
        is_do = jnp.equal(action, _DO_ACTION)

        def _kills(prev_pos, prev_mask, new_mask):
            in_facing = jnp.all(prev_pos == facing_tile, axis=-1)
            killed = prev_mask & jnp.logical_not(new_mask) & in_facing & is_do
            return jnp.sum(killed).astype(jnp.float32)

        zombie_kills = _kills(
            prev.zombies.position, state.prev_zombie_mask, env_state.zombies.mask
        )
        skeleton_kills = _kills(
            prev.skeletons.position, state.prev_skeleton_mask, env_state.skeletons.mask
        )

        not_done = jnp.logical_not(jnp.asarray(done, dtype=jnp.bool_)).astype(jnp.float32)
        combat_reward = (zombie_kills + skeleton_kills) * self.alpha * not_done

        info["base_reward"] = base_reward
        info["combat_reward"] = combat_reward
        info["combat_zombie_kills"] = zombie_kills
        info["combat_skeleton_kills"] = skeleton_kills

        new_state = CombatEnvState(
            env_state=env_state,
            prev_zombie_mask=env_state.zombies.mask,
            prev_skeleton_mask=env_state.skeletons.mask,
        )

        reward = base_reward + combat_reward
        return obs, new_state, reward, done, info
