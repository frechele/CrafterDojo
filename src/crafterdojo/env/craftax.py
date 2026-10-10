import hashlib

import cv2
import gymnasium as gym
import gymnasium.spaces as spaces
import jax
import jax.numpy as jnp
import numpy as np
from craftax.craftax_classic.constants import Achievement, Action, BlockType, SOLID_BLOCKS
from craftax.craftax_classic.renderer import render_craftax_pixels
from craftax.craftax_env import make_craftax_env_from_name

from crafterdojo.env import TASKS
from crafterdojo.env.wrappers import (
    ActionTrackingWrapper,
    AchievementWrapper,
    EvalTrackingWrapper,
    SummaryTrackingWrapper,
    StateTrackingWrapper,
    VPTWrapper,
)


AGENT_RESOLUTION = (144, 144)
CRAFTAX_CLASSIC_PIXELS_ENV_NAME = "Craftax-Classic-Pixels-v1"
ACHIEVEMENT_NAMES = tuple(achievement.name.lower() for achievement in Achievement)
ACHIEVEMENT_INDEX = {
    achievement.name.lower(): achievement.value for achievement in Achievement
}
ACTION_NAMES = tuple(action.name.lower() for action in Action)
ACTION_INDEX = {name: idx for idx, name in enumerate(ACTION_NAMES)}
CRAFTER_ACTION_ALIASES = {
    "move_left": "left",
    "move_right": "right",
    "move_up": "up",
    "move_down": "down",
}
SUPPORTED_DOJO_TASK_TYPES = {
    "achievement",
    "harvest",
    "conditional",
    "conditional_3",
    "survival",
}
RENDER_CELL_SIZE = 16


def _make_seed(base_seed: int | None, episode_idx: int) -> int:
    if base_seed is None:
        return episode_idx

    digest = hashlib.sha256(f"{base_seed}:{episode_idx}".encode("ascii")).digest()
    return int.from_bytes(digest[:4], "little")


def _render_state(state) -> np.ndarray:
    frame = np.asarray(render_craftax_pixels(state, RENDER_CELL_SIZE)).astype(np.uint8)
    if frame.shape[:2] != AGENT_RESOLUTION[::-1]:
        frame = cv2.resize(frame, AGENT_RESOLUTION, interpolation=cv2.INTER_LINEAR)
    return frame


def _normalize_action(action) -> int:
    if isinstance(action, str):
        action_name = CRAFTER_ACTION_ALIASES.get(action, action)
        if action_name not in ACTION_INDEX:
            raise ValueError(f"Unknown action '{action}'.")
        return ACTION_INDEX[action_name]

    if isinstance(action, np.ndarray) and action.shape == ():
        return int(action.item())

    if hasattr(action, "item") and not isinstance(action, (list, tuple, dict)):
        try:
            return int(action.item())
        except (TypeError, ValueError):
            pass

    return int(action)


def _inventory_to_dict(inventory) -> dict[str, int]:
    return {
        "wood": int(inventory.wood),
        "stone": int(inventory.stone),
        "coal": int(inventory.coal),
        "iron": int(inventory.iron),
        "diamond": int(inventory.diamond),
        "sapling": int(inventory.sapling),
        "wood_pickaxe": int(inventory.wood_pickaxe),
        "stone_pickaxe": int(inventory.stone_pickaxe),
        "iron_pickaxe": int(inventory.iron_pickaxe),
        "wood_sword": int(inventory.wood_sword),
        "stone_sword": int(inventory.stone_sword),
        "iron_sword": int(inventory.iron_sword),
    }


def _state_inventory_to_dict(state) -> dict[str, int]:
    inventory = _inventory_to_dict(state.inventory)
    inventory.update(
        {
            "health": int(np.asarray(state.player_health)),
            "food": int(np.asarray(state.player_food)),
            "drink": int(np.asarray(state.player_drink)),
            "energy": int(np.asarray(state.player_energy)),
        }
    )
    return inventory


def _achievements_to_dict(achievements) -> dict[str, int]:
    achievements_np = np.asarray(achievements)
    return {
        name: int(achievements_np[idx]) for idx, name in enumerate(ACHIEVEMENT_NAMES)
    }


def _resolve_task(task_name: str) -> dict[str, object]:
    if task_name not in TASKS:
        raise ValueError(f"Unknown CrafterDojo task '{task_name}'.")

    task = TASKS[task_name].copy()
    task_type = task["task_type"]
    if task_type not in SUPPORTED_DOJO_TASK_TYPES:
        raise ValueError(
            f"Unsupported building-style task '{task_name}' with task type '{task_type}'."
        )

    task["name"] = task_name
    return task


def _achievement_value(state, achievement_name: str) -> int:
    return int(np.asarray(state.achievements[ACHIEVEMENT_INDEX[achievement_name]]))


def _get_state_item_count(state, item_name: str) -> int:
    if hasattr(state.inventory, item_name):
        return int(np.asarray(getattr(state.inventory, item_name)))

    special_counts = {
        "health": state.player_health,
        "food": state.player_food,
        "drink": state.player_drink,
        "energy": state.player_energy,
    }
    if item_name in special_counts:
        return int(np.asarray(special_counts[item_name]))

    raise KeyError(f"Unsupported inventory item '{item_name}'.")


def _set_state_item_count(state, item_name: str, value: int):
    if hasattr(state.inventory, item_name):
        inventory = state.inventory.replace(**{item_name: int(value)})
        return state.replace(inventory=inventory)

    special_updates = {
        "health": "player_health",
        "food": "player_food",
        "drink": "player_drink",
        "energy": "player_energy",
    }
    if item_name in special_updates:
        return state.replace(**{special_updates[item_name]: int(value)})

    raise KeyError(f"Unsupported inventory item '{item_name}'.")


def _clear_mobs(mobs):
    return mobs.replace(
        mask=jnp.zeros_like(mobs.mask, dtype=bool),
        attack_cooldown=jnp.zeros_like(mobs.attack_cooldown),
    )


def _rebuild_mob_map(state):
    mob_map = jnp.zeros_like(state.mob_map, dtype=bool)
    for mobs in (state.zombies, state.cows, state.skeletons, state.arrows):
        positions = np.asarray(mobs.position)
        mask = np.asarray(mobs.mask, dtype=bool)
        for x, y in positions[mask]:
            mob_map = mob_map.at[int(x), int(y)].set(True)
    return state.replace(mob_map=mob_map)


def _remove_hostiles(state):
    state = state.replace(
        zombies=_clear_mobs(state.zombies),
        skeletons=_clear_mobs(state.skeletons),
        arrows=_clear_mobs(state.arrows),
    )
    return _rebuild_mob_map(state)


def _remove_cows(state):
    state = state.replace(cows=_clear_mobs(state.cows))
    return _rebuild_mob_map(state)


# Per-mob defaults: (state attribute name, default health on spawn).
MOB_DEFAULTS = {
    "zombie": ("zombies", 5),
    "skeleton": ("skeletons", 3),
}

_SOLID_BLOCK_VALUES = frozenset(int(v) for v in np.asarray(SOLID_BLOCKS))


def _candidate_spawn_tiles(state, dist_min: int, dist_max: int) -> list[tuple[int, int]]:
    px, py = (int(v) for v in np.asarray(state.player_position, dtype=np.int32))
    width, height = state.map.shape
    map_np = np.asarray(state.map, dtype=np.int32)
    mob_map = np.asarray(state.mob_map, dtype=bool)
    lava_value = BlockType.LAVA.value

    out: list[tuple[int, int]] = []
    for dx in range(-dist_max, dist_max + 1):
        for dy in range(-dist_max, dist_max + 1):
            md = abs(dx) + abs(dy)
            if md < dist_min or md > dist_max:
                continue
            x, y = px + dx, py + dy
            if not (0 <= x < width and 0 <= y < height):
                continue
            if (x, y) == (px, py):
                continue
            if mob_map[x, y]:
                continue
            block = int(map_np[x, y])
            if block in _SOLID_BLOCK_VALUES or block == lava_value:
                continue
            out.append((x, y))
    return out


def _normalize_dist(dist) -> tuple[int, int]:
    if isinstance(dist, int):
        return dist, dist
    pair = tuple(int(v) for v in dist)
    if len(pair) != 2:
        raise ValueError(f"spawn_mobs.dist must be int or [min, max]; got {dist!r}")
    lo, hi = pair
    if lo > hi:
        raise ValueError(f"spawn_mobs.dist min ({lo}) > max ({hi})")
    if lo < 1:
        raise ValueError(f"spawn_mobs.dist min must be ≥ 1 (got {lo}) — distance 0 is the player tile")
    return lo, hi


def _force_spawn_mobs(state, spec: dict, rng):
    """Place mobs on free tiles around the player.

    `spec` is `{mob_type: {count, dist}}`. Distances are manhattan, inclusive.
    Each placement consumes a free slot in the matching state mob array; if no
    free slot is left or no candidate tile is in range, the placement is silently
    skipped (capacity is small: max_zombies=3, max_skeletons=2 by default).
    """
    actual = {mob_type: 0 for mob_type in spec}
    for mob_type, cfg in spec.items():
        if mob_type not in MOB_DEFAULTS:
            raise ValueError(
                f"Unsupported spawn_mobs type '{mob_type}'. "
                f"Allowed: {sorted(MOB_DEFAULTS.keys())}"
            )
        attr, default_hp = MOB_DEFAULTS[mob_type]
        count = int(cfg["count"])
        dist_min, dist_max = _normalize_dist(cfg["dist"])

        for _ in range(count):
            mobs = getattr(state, attr)
            mask = np.asarray(mobs.mask, dtype=bool)
            free_slots = np.where(~mask)[0]
            if len(free_slots) == 0:
                break
            cands = _candidate_spawn_tiles(state, dist_min, dist_max)
            if not cands:
                break
            rng, sub = jax.random.split(rng)
            pick_idx = int(jax.random.randint(sub, (), 0, len(cands)))
            tx, ty = cands[pick_idx]
            slot = int(free_slots[0])

            new_mobs = mobs.replace(
                position=mobs.position.at[slot].set(jnp.array([tx, ty], dtype=jnp.int32)),
                mask=mobs.mask.at[slot].set(True),
                health=mobs.health.at[slot].set(jnp.int32(default_hp)),
                attack_cooldown=mobs.attack_cooldown.at[slot].set(jnp.int32(0)),
            )
            state = state.replace(**{attr: new_mobs})
            state = _rebuild_mob_map(state)
            actual[mob_type] += 1

    return state, actual


def _nearby_info(state) -> dict[str, bool]:
    px, py = np.asarray(state.player_position, dtype=np.int32)
    width, height = state.map.shape

    nearby_table = False
    nearby_furnace = False
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            x = int(px + dx)
            y = int(py + dy)
            if x < 0 or x >= width or y < 0 or y >= height:
                continue

            block = int(np.asarray(state.map[x, y]))
            nearby_table = nearby_table or block == BlockType.CRAFTING_TABLE.value
            nearby_furnace = nearby_furnace or block == BlockType.FURNACE.value

    return {
        "nearby_table": nearby_table,
        "nearby_furnace": nearby_furnace,
    }


def _is_in_lava(state) -> bool:
    x, y = np.asarray(state.player_position, dtype=np.int32)
    return int(np.asarray(state.map[int(x), int(y)])) == BlockType.LAVA.value


def _is_game_over(state, params) -> bool:
    timestep = int(np.asarray(state.timestep))
    health = int(np.asarray(state.player_health))
    return timestep >= int(params.max_timesteps) or health <= 0 or _is_in_lava(state)


class CraftaxClassicTaskLogic:
    def __init__(self, task: str):
        self.task = task
        self.task_spec = _resolve_task(task)
        self._on_reset()

    def reset(self, state):
        del state
        self._on_reset()

    def transition(self, prev_state, cur_state) -> tuple[float, bool | None]:
        success = self._check_success(prev_state, cur_state)
        reward = self._compute_reward(prev_state, cur_state, success)
        return reward, success

    def _on_reset(self):
        self._second_at_first_done = None
        self._third_at_second_done = None
        self._phase = 0
        self._is_first_for_first = True

    def _check_success(self, prev_state, cur_state) -> bool | None:
        task_type = self.task_spec["task_type"]

        if task_type == "achievement":
            if _achievement_value(cur_state, self.task_spec["achievement"]) > 0:
                return True
            return None

        if task_type == "harvest":
            target_item = self.task_spec["target_item"]
            if _get_state_item_count(cur_state, target_item) > _get_state_item_count(
                prev_state, target_item
            ):
                return True
            return None

        if task_type == "conditional":
            first_done = _achievement_value(cur_state, self.task_spec["first"]) > 0
            second_done = _achievement_value(cur_state, self.task_spec["second"])

            if self._is_first_for_first and first_done:
                self._second_at_first_done = second_done
                self._is_first_for_first = False

            if (
                first_done
                and self._second_at_first_done is not None
                and second_done > self._second_at_first_done
            ):
                return True

            return None

        if task_type == "conditional_3":
            first_done = _achievement_value(cur_state, self.task_spec["first"])
            second_done = _achievement_value(cur_state, self.task_spec["second"])
            third_done = _achievement_value(cur_state, self.task_spec["third"])

            if self._phase == 0 and first_done > 0:
                self._second_at_first_done = second_done
                self._phase = 1

            if (
                self._phase == 1
                and self._second_at_first_done is not None
                and second_done > self._second_at_first_done
            ):
                self._third_at_second_done = third_done
                self._phase = 2

            if (
                self._phase == 2
                and self._third_at_second_done is not None
                and third_done > self._third_at_second_done
            ):
                return True

            return None

        if task_type == "survival":
            return None

        raise ValueError(f"Unsupported task type '{task_type}'.")

    def _compute_reward(self, prev_state, cur_state, success: bool | None) -> float:
        if self.task_spec["task_type"] == "survival":
            prev_light = float(np.asarray(prev_state.light_level))
            cur_light = float(np.asarray(cur_state.light_level))
            return float(prev_light < 0.5 and cur_light >= 0.5)

        return 1.0 if success else 0.0


class CraftaxClassicPixelsAdapter(gym.Env):
    """Gymnasium-compatible adapter over Craftax-Classic pixel observations."""

    metadata = {}
    reward_range = None

    def __init__(self, seed: int | None = None):
        super().__init__()
        self._env = make_craftax_env_from_name(CRAFTAX_CLASSIC_PIXELS_ENV_NAME, False)
        self._params = self._env.default_params
        self._base_seed = seed
        self._episode_idx = 0
        self._state = None
        self._key = None

        self.observation_space = spaces.Box(
            0, 255, shape=(AGENT_RESOLUTION[1], AGENT_RESOLUTION[0], 3), dtype=np.uint8
        )
        self.action_space = spaces.Discrete(self._env.action_space(self._params).n)

    @property
    def action_names(self):
        return tuple(range(self.action_space.n))

    def reset(self, *, seed=None, options=None):
        del options

        episode_seed = _make_seed(
            self._base_seed if seed is None else seed, self._episode_idx
        )
        self._episode_idx += 1

        self._key = jax.random.PRNGKey(episode_seed)
        _, self._state = self._env.reset(self._key, self._params)

        return _render_state(self._state), self._build_info(0.0)

    def step(self, action):
        if self._state is None:
            raise RuntimeError("Environment must be reset before stepping.")

        action = _normalize_action(action)
        self._key, step_key = jax.random.split(self._key)
        _, self._state, reward, done, info = self._env.step(
            step_key, self._state, action, self._params
        )

        reward_value = float(np.asarray(reward))
        done_value = bool(np.asarray(done))
        merged_info = self._build_info(reward_value)
        merged_info["discount"] = float(np.asarray(info.get("discount", 1.0)))
        merged_info["score"] = float(np.asarray(info.get("score", 0.0)))

        return _render_state(self._state), reward_value, done_value, merged_info

    def close(self):
        self._state = None

    def _build_info(self, reward: float) -> dict[str, object]:
        return {
            "inventory": _inventory_to_dict(self._state.inventory),
            "achievements": _achievements_to_dict(self._state.achievements),
            "player_pos": np.asarray(self._state.player_position, dtype=np.int32),
            "reward": reward,
        }


class CraftaxClassicDojoEnv(gym.Env):
    """Task-oriented Craftax-Classic environment with CrafterDojo success checks."""

    metadata = {}
    reward_range = None

    def __init__(
        self,
        task: str,
        seed: int | None = None,
    ):
        super().__init__()

        self._env = make_craftax_env_from_name(CRAFTAX_CLASSIC_PIXELS_ENV_NAME, False)
        self._task = _resolve_task(task)
        self._task_logic = CraftaxClassicTaskLogic(task)

        self._base_seed = seed
        self._episode_idx = 0
        self._state = None
        self._key = None
        self._spawn_counts: dict[str, int] = {}

        self._params = self._build_params()

        self.observation_space = spaces.Box(
            0, 255, shape=(AGENT_RESOLUTION[1], AGENT_RESOLUTION[0], 3), dtype=np.uint8
        )
        self.action_space = spaces.Discrete(self._env.action_space(self._params).n)

    @property
    def action_names(self):
        return ACTION_NAMES

    @property
    def target_item(self) -> str | None:
        return self._task.get("target_item")

    @property
    def task_spec(self) -> dict:
        return self._task

    def reset(self, *, seed=None, options=None):
        del options

        if seed is None:
            episode_seed = _make_seed(self._base_seed, self._episode_idx)
            self._episode_idx += 1
        else:
            episode_seed = _make_seed(seed, 0)

        self._key = jax.random.PRNGKey(episode_seed)
        self._key, spawn_key = jax.random.split(self._key)
        _, self._state = self._env.reset(self._key, self._params)
        self._state = self._apply_reset_overrides(self._state, spawn_key)
        self._task_logic.reset(self._state)

        return _render_state(self._state), self._build_info(0.0)

    def step(self, action):
        if self._state is None:
            raise RuntimeError("Environment must be reset before stepping.")

        prev_state = self._state
        action = _normalize_action(action)
        self._key, step_key = jax.random.split(self._key)
        _, next_state, _, _, _ = self._env.step(
            step_key, prev_state, action, self._params
        )

        next_state = self._apply_step_overrides(prev_state, next_state)
        reward, success = self._task_logic.transition(prev_state, next_state)
        done = _is_game_over(next_state, self._params) or success is not None

        self._state = next_state
        info = self._build_info(reward)
        info["discount"] = (
            0.0 if int(np.asarray(next_state.player_health)) <= 0 else 1.0
        )
        if done:
            info["is_success"] = bool(success)
            info["success"] = bool(success)

        return _render_state(self._state), reward, done, info

    def render(self):
        if self._state is None:
            raise RuntimeError("Environment must be reset before rendering.")
        return _render_state(self._state)

    def close(self):
        self._state = None
        self._key = None

    def _build_params(self):
        params = self._env.default_params
        length = self._task.get("length", params.max_timesteps)
        params = params.replace(max_timesteps=int(length))

        if self._task.get("peaceful", False) or self._task.get(
            "no_extra_hostile_spawn", False
        ):
            params = params.replace(
                spawn_zombie_base_chance=0.0,
                spawn_zombie_night_chance=0.0,
                spawn_skeleton_chance=0.0,
            )

        if self._task.get("no_cow", False):
            params = params.replace(spawn_cow_chance=0.0)

        return params

    def _apply_reset_overrides(self, state, spawn_rng):
        initial_inventory = self._task.get("initial_inventory", {}) or {}
        for item_name, value in initial_inventory.items():
            state = _set_state_item_count(state, item_name, value)

        if self._task.get("peaceful", False):
            state = _remove_hostiles(state)

        if self._task.get("no_cow", False):
            state = _remove_cows(state)

        spawn_mobs = self._task.get("spawn_mobs") or {}
        if spawn_mobs:
            state, self._spawn_counts = _force_spawn_mobs(state, spawn_mobs, spawn_rng)
        else:
            self._spawn_counts = {}

        return state

    def _apply_step_overrides(self, prev_state, state):
        if self._task.get("easy_survive", False) and not _is_in_lava(state):
            state = state.replace(
                player_health=max(
                    int(np.asarray(state.player_health)),
                    int(np.asarray(prev_state.player_health)),
                )
            )

        if self._task.get("peaceful", False):
            state = _remove_hostiles(state)

        if self._task.get("no_cow", False):
            state = _remove_cows(state)

        return state

    def _build_info(self, reward: float) -> dict[str, object]:
        info = {
            "inventory": _state_inventory_to_dict(self._state),
            "achievements": _achievements_to_dict(self._state.achievements),
            "player_pos": np.asarray(self._state.player_position, dtype=np.int32),
            "semantic": np.asarray(self._state.map, dtype=np.int32),
            "zombies_position": np.asarray(self._state.zombies.position, dtype=np.int32),
            "zombies_mask": np.asarray(self._state.zombies.mask, dtype=bool),
            "skeletons_position": np.asarray(
                self._state.skeletons.position, dtype=np.int32
            ),
            "skeletons_mask": np.asarray(self._state.skeletons.mask, dtype=bool),
            "arrows_position": np.asarray(self._state.arrows.position, dtype=np.int32),
            "arrows_mask": np.asarray(self._state.arrows.mask, dtype=bool),
            "target_item": self._task.get("target_item"),
            "reward": reward,
        }
        info.update(_nearby_info(self._state))
        if self._spawn_counts:
            info["spawned_mobs"] = dict(self._spawn_counts)
        return info


def craftax_classic_pixels_main(
    config, logdir: str, train: bool, fabric, episode_id=None
):
    del config, fabric

    env = CraftaxClassicPixelsAdapter(seed=episode_id)

    if not train:
        env = StateTrackingWrapper(env, logdir, episode_id)
        env = EvalTrackingWrapper(env, logdir, episode_id)
        env = ActionTrackingWrapper(env, logdir, episode_id)
        env = AchievementWrapper(env, logdir, episode_id)

    env = VPTWrapper(env)
    return env


def craftax_classic_dojo_main(
    config, logdir: str, train: bool, fabric, episode_id=None
):
    del fabric

    task = config["task"]
    if task is None:
        raise ValueError("craftax_classic_dojo env requires a non-null task.")

    env = CraftaxClassicDojoEnv(
        task=task,
        seed=episode_id,
    )

    if not train:
        if config.get("summary_only", False):
            env = SummaryTrackingWrapper(env, logdir, episode_id)
        else:
            env = StateTrackingWrapper(env, logdir, episode_id)
            env = EvalTrackingWrapper(env, logdir, episode_id)
            env = ActionTrackingWrapper(env, logdir, episode_id)
            env = AchievementWrapper(env, logdir, episode_id)

    env = VPTWrapper(env)
    return env
