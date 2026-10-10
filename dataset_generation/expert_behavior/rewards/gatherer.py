from functools import partial
from typing import Any, Union

import chex
import jax
import jax.numpy as jnp
from flax import struct

from dataset_generation.expert_behavior.rewards.base import RewardWrapper
from dataset_generation.expert_behavior.rewards.registry import register_reward


# Inventory dataclass field order in Craftax-Classic:
# wood=0, stone=1, coal=2, iron=3, diamond=4, sapling=5,
# wood_pickaxe=6, stone_pickaxe=7, iron_pickaxe=8,
# wood_sword=9, stone_sword=10, iron_sword=11.
_WOOD_IDX = 0
_STONE_IDX = 1
_SAPLING_IDX = 5

# Craftax-Classic Action values
# (see craftax.craftax_classic.constants.Action).
_PLACE_STONE_ACTION = 7
_PLACE_TABLE_ACTION = 8
_PLACE_FURNACE_ACTION = 9
_PLACE_PLANT_ACTION = 10


@struct.dataclass
class GathererEnvState:
    env_state: Any
    max_inventory_flat: jnp.ndarray  # kept for stone_place_penalty (legacy) only
    prev_inv_flat: jnp.ndarray

    def __getattr__(self, name):
        return getattr(self.env_state, name)


def _inventory_to_array(inventory) -> jnp.ndarray:
    """Convert an Inventory struct to a flat jnp array."""
    return jnp.array(jax.tree_util.tree_leaves(inventory), dtype=jnp.float32)


@register_reward("gatherer")
class GathererRewardWrapper(RewardWrapper):
    """Reward agent for inventory acquisition, with fake-work placement penalty.

    Default (``place_penalty=True``): every per-step positive inventory delta
    earns ``+alpha`` per unit (all items — wood, stone, coal, iron, diamond,
    sapling, pickaxes, swords). Placement actions that consume inventory
    without producing an item incur an offsetting ``-alpha`` penalty per unit
    lost, so place-and-re-collect cycles sum to zero:

      * stone loss penalized on PLACE_STONE or PLACE_FURNACE
      * wood loss penalized on PLACE_TABLE (consumes 2 wood per table)
      * sapling loss penalized on PLACE_PLANT

    Crafting-induced consumption (MAKE_STONE_PICKAXE, MAKE_IRON_SWORD, etc.)
    is NOT penalized: the consumed raw materials simply convert into a
    rewardable tool, and tool accumulation is an intended incentive.

    ``penalize_plant`` (default True) controls whether sapling loss on
    PLACE_PLANT incurs a penalty. PLACE_PLANT has no near-term inventory
    payoff (unlike PLACE_TABLE/FURNACE which enable immediate crafting
    rewards), so the constant penalty may cause under-planting. Setting
    ``penalize_plant=False`` lets the agent plant freely.

    ``stone_place_penalty`` is the legacy single-action variant that only
    penalizes stone loss via PLACE_STONE; PLACE_FURNACE can still farm
    reward via consume-then-remine cycles. Kept for backward compatibility
    of older datasets; prefer ``place_penalty`` for new training.
    """

    def __init__(
        self,
        env,
        alpha: float = 0.1,
        stone_place_penalty: bool = False,
        place_penalty: bool = False,
        penalize_plant: bool = True,
    ):
        super().__init__(env)
        if stone_place_penalty and place_penalty:
            raise ValueError(
                "stone_place_penalty and place_penalty are mutually exclusive"
            )
        self.alpha = alpha
        self.stone_place_penalty = bool(stone_place_penalty)
        self.place_penalty = bool(place_penalty)
        self.penalize_plant = bool(penalize_plant)

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, key: chex.PRNGKey, params=None):
        obs, env_state = self._env.reset(key, params)
        inv_flat = _inventory_to_array(env_state.inventory)
        state = GathererEnvState(
            env_state=env_state,
            max_inventory_flat=inv_flat,
            prev_inv_flat=inv_flat,
        )
        return obs, state

    @partial(jax.jit, static_argnums=(0, 4))
    def step(self, key: chex.PRNGKey, state, action: Union[int, float], params=None):
        obs, env_state, base_reward, done, info = self._env.step(
            key, state.env_state, action, params
        )

        curr_inv = _inventory_to_array(env_state.inventory)

        if self.place_penalty:
            inv_delta = curr_inv - state.prev_inv_flat
            gain = jnp.maximum(inv_delta, 0.0)
            loss = jnp.minimum(inv_delta, 0.0)

            is_place_stone = jnp.asarray(
                action == _PLACE_STONE_ACTION, dtype=jnp.float32
            )
            is_place_table = jnp.asarray(
                action == _PLACE_TABLE_ACTION, dtype=jnp.float32
            )
            is_place_furnace = jnp.asarray(
                action == _PLACE_FURNACE_ACTION, dtype=jnp.float32
            )
            is_place_plant = jnp.asarray(
                action == _PLACE_PLANT_ACTION, dtype=jnp.float32
            )

            plant_coef = 1.0 if self.penalize_plant else 0.0
            placement_penalty = (
                loss[_STONE_IDX] * (is_place_stone + is_place_furnace)
                + loss[_WOOD_IDX] * is_place_table
                + loss[_SAPLING_IDX] * is_place_plant * plant_coef
            )

            gatherer_reward = (jnp.sum(gain) + placement_penalty) * self.alpha

            new_max = state.max_inventory_flat
            info["gatherer_stone_delta"] = inv_delta[_STONE_IDX]
            info["gatherer_wood_delta"] = inv_delta[_WOOD_IDX]
            info["gatherer_sapling_delta"] = inv_delta[_SAPLING_IDX]
            info["gatherer_place_penalty"] = placement_penalty * self.alpha
        elif self.stone_place_penalty:
            peak_gains = jnp.maximum(curr_inv - state.max_inventory_flat, 0.0)
            new_max = jnp.maximum(state.max_inventory_flat, curr_inv)
            peak_gains = peak_gains.at[_STONE_IDX].set(0.0)
            stone_delta = curr_inv[_STONE_IDX] - state.prev_inv_flat[_STONE_IDX]
            stone_gain = jnp.maximum(stone_delta, 0.0)
            stone_loss = jnp.minimum(stone_delta, 0.0)
            is_place_stone = jnp.asarray(
                action == _PLACE_STONE_ACTION, dtype=jnp.float32
            )
            stone_component = stone_gain + stone_loss * is_place_stone
            gatherer_reward = (jnp.sum(peak_gains) + stone_component) * self.alpha
            info["gatherer_stone_delta"] = stone_delta
            info["gatherer_stone_place_penalty"] = (
                stone_loss * is_place_stone * self.alpha
            )
        else:
            peak_gains = jnp.maximum(curr_inv - state.max_inventory_flat, 0.0)
            new_max = jnp.maximum(state.max_inventory_flat, curr_inv)
            gatherer_reward = jnp.sum(peak_gains) * self.alpha

        new_state = GathererEnvState(
            env_state=env_state,
            max_inventory_flat=new_max,
            prev_inv_flat=curr_inv,
        )

        info["base_reward"] = base_reward
        info["gatherer_reward"] = gatherer_reward

        reward = base_reward + gatherer_reward
        return obs, new_state, reward, done, info
