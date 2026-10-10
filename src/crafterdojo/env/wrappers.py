import copy
import cv2
import json
import numpy as np
import os
import pickle
import torch
import gymnasium.spaces as spaces
from gymnasium import Wrapper
from gymnasium.utils.step_api_compatibility import convert_to_terminated_truncated_step_api


def _propagate_begin_episode(inner_env, episode_id: int):
    target = inner_env
    if hasattr(target, "begin_episode"):
        target.begin_episode(episode_id)


def _json_ready(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.ndim == 0:
            return value.item()
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


class VPTWrapper(Wrapper):
    def __init__(self, env):
        super(VPTWrapper, self).__init__(env)

        obs_space = env.observation_space
        height, width = obs_space.shape[:2]

        self.observation_space = spaces.Dict(
            {"img": spaces.Box(0, 255, shape=(height, width, 3), dtype=np.uint8)}
        )

    def begin_episode(self, episode_id: int):
        _propagate_begin_episode(self.env, episode_id)

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        return self._get_obs(obs), info

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self._get_obs(obs)
        return obs, reward, done, info

    def _get_obs(self, obs):
        return {"img": obs}


class DeathPenaltyWrapper(Wrapper):
    def __init__(self, env, live_bonus: float, death_penalty: int):
        super(DeathPenaltyWrapper, self).__init__(env)
        self.live_bonus = live_bonus
        self.death_penalty = death_penalty

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        is_dead = info["inventory"]["health"] <= 0
        if is_dead:
            reward -= self.death_penalty
        else:
            reward += self.live_bonus

        return obs, reward, done, info


class BaseEvalTrackingWrapper(Wrapper):
    def __init__(self, env, logdir: str, episode_id: int | None, **kwargs):
        super(BaseEvalTrackingWrapper, self).__init__(env)

        self.episode_path = os.path.join(logdir, "episode")
        self.summary_path = os.path.join(logdir, "summary")
        os.makedirs(self.episode_path, exist_ok=True)
        os.makedirs(self.summary_path, exist_ok=True)

        self.episode_id = episode_id
        self._reset_episode_buffers()

    def _reset_episode_buffers(self):
        self.obs_log = []
        self.log = dict(reward=[])
        self.last_info = None
        self._saved = False

    def begin_episode(self, episode_id: int):
        if self.episode_id is not None and not self._saved:
            self._save_tracking()
        _propagate_begin_episode(self.env, episode_id)
        self.episode_id = episode_id
        self._reset_episode_buffers()

    def close(self):
        if self.episode_id is not None and not self._saved:
            self._save_tracking()
            self._saved = True
        super().close()

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)

        self._update_from_obs(obs)
        self._update_after_reset()

        return obs, info

    def step(self, action, **kwargs):
        obs, reward, done, info = self.env.step(action, **kwargs)

        self._update_from_obs(obs)
        self.log["reward"].append(reward)
        self._update_from_info(info)
        self.last_info = copy.deepcopy(info)

        return obs, reward, done, info

    def _update_from_obs(self, obs):
        self.obs_log.append(obs)

    def _update_from_info(self, info):
        pass

    def _update_after_reset(self):
        pass

    def _save_tracking(self):
        self.log["obs"] = self.obs_log
        np.savez_compressed(
            os.path.join(self.episode_path, f"episode_{self.episode_id:04d}.npz"),
            **self.log,
        )
        self._save_summary()
        self._saved = True

    def _save_summary(self):
        final_info = self.last_info or {}
        rewards = self.log["reward"]
        summary = {
            "episode_id": int(self.episode_id),
            "episode_length": int(len(rewards)),
            "total_reward": float(np.sum(rewards, dtype=np.float64)) if rewards else 0.0,
            "success": final_info.get("success", final_info.get("is_success")),
            "achievements": _json_ready(final_info.get("achievements", {})),
            "inventory": _json_ready(final_info.get("inventory", {})),
            "final_info": _json_ready(final_info),
        }
        with open(
            os.path.join(self.summary_path, f"episode_{self.episode_id:04d}.json"), "wt"
        ) as f:
            json.dump(summary, f, indent=2, sort_keys=True)


class EvalTrackingWrapper(BaseEvalTrackingWrapper):
    def __init__(self, env, logdir: str, episode_id: int | None, **kwargs):
        super(EvalTrackingWrapper, self).__init__(env, logdir, episode_id, **kwargs)

        video_shape = self.env.observation_space.shape

        self.video_path = os.path.join(logdir, "video")
        os.makedirs(self.video_path, exist_ok=True)

        self.info_path = os.path.join(logdir, "info")
        os.makedirs(self.info_path, exist_ok=True)

        self.video_shape = video_shape
        self.infos = []
        self.writer = None
        if self.episode_id is not None:
            self._open_writer()

    def _open_writer(self):
        self.writer = cv2.VideoWriter(
            os.path.join(self.video_path, f"episode_{self.episode_id:04d}.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            10,
            (self.video_shape[1], self.video_shape[0]),
        )

    def _reset_episode_buffers(self):
        super()._reset_episode_buffers()
        self.infos = []
        self.writer = None

    def begin_episode(self, episode_id: int):
        super().begin_episode(episode_id)
        self._open_writer()

    def _update_from_obs(self, obs):
        frame = obs.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self.writer.write(frame)

    def _update_from_info(self, info):
        self.infos.append(info.copy())

    def _save_tracking(self):
        if self.writer is not None:
            self.writer.release()
        np.savez_compressed(
            os.path.join(self.episode_path, f"episode_{self.episode_id:04d}.npz"),
            **self.log,
        )

        with open(
            os.path.join(self.info_path, f"episode_{self.episode_id:04d}.pkl"), "wb"
        ) as f:
            pickle.dump(self.infos, f)
        self._save_summary()
        self._saved = True


class SummaryTrackingWrapper(Wrapper):
    def __init__(self, env, logdir: str, episode_id: int | None):
        super(SummaryTrackingWrapper, self).__init__(env)
        self.summary_path = os.path.join(logdir, "summary")
        os.makedirs(self.summary_path, exist_ok=True)
        self.episode_id = episode_id
        self.rewards = []
        self.last_info = None
        self._saved = False

    def begin_episode(self, episode_id: int):
        if self.episode_id is not None and not self._saved:
            self._save_summary()
        _propagate_begin_episode(self.env, episode_id)
        self.episode_id = episode_id
        self.rewards = []
        self.last_info = None
        self._saved = False

    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.rewards.append(reward)
        self.last_info = copy.deepcopy(info)
        return obs, reward, done, info

    def close(self):
        if self.episode_id is not None and not self._saved:
            self._save_summary()
            self._saved = True
        self.env.close()

    def _save_summary(self):
        final_info = self.last_info or {}
        summary = {
            "episode_id": int(self.episode_id),
            "episode_length": int(len(self.rewards)),
            "total_reward": (
                float(np.sum(self.rewards, dtype=np.float64)) if self.rewards else 0.0
            ),
            "success": final_info.get("success", final_info.get("is_success")),
            "achievements": _json_ready(final_info.get("achievements", {})),
            "inventory": _json_ready(final_info.get("inventory", {})),
            "final_info": _json_ready(final_info),
        }
        with open(
            os.path.join(self.summary_path, f"episode_{self.episode_id:04d}.json"), "wt"
        ) as f:
            json.dump(summary, f, indent=2, sort_keys=True)
        self._saved = True


class AchievementWrapper(Wrapper):
    def __init__(self, env, logdir: str, episode_id: int | None):
        super(AchievementWrapper, self).__init__(env)

        self.achievement_path = os.path.join(logdir, "achievements")
        os.makedirs(self.achievement_path, exist_ok=True)

        self.episode_id = episode_id
        self.achievements = None
        self._saved = False

    def begin_episode(self, episode_id: int):
        if self.episode_id is not None and not self._saved:
            self._save_achievements()
        _propagate_begin_episode(self.env, episode_id)
        self.episode_id = episode_id
        self.achievements = None
        self._saved = False

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.achievements = info["achievements"]

        return obs, reward, done, info

    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)

    def close(self):
        if self.episode_id is not None and not self._saved:
            self._save_achievements()
            self._saved = True
        self.env.close()

    def _save_achievements(self):
        filepath = os.path.join(
            self.achievement_path, f"episode_{self.episode_id:04d}.json"
        )
        with open(filepath, "wt") as f:
            json.dump(self.achievements, f)
        self._saved = True


class ActionTrackingWrapper(Wrapper):
    def __init__(self, env, logdir: str, episode_id: int | None, **kwargs):
        super(ActionTrackingWrapper, self).__init__(env)

        self.episode_id = episode_id
        self.action_path = os.path.join(logdir, "action")
        os.makedirs(self.action_path, exist_ok=True)
        self.action_log = []
        self._saved = False

    def __del__(self):
        try:
            self._save_action_log()
        except Exception:
            pass

    def begin_episode(self, episode_id: int):
        if self.episode_id is not None and not self._saved:
            self._save_action_log()
        _propagate_begin_episode(self.env, episode_id)
        self.episode_id = episode_id
        self.action_log = []
        self._saved = False

    def close(self):
        if self.episode_id is not None and not self._saved:
            self._save_action_log()
        super().close()

    def _save_action_log(self):
        if self._saved:
            return
        with open(
            os.path.join(self.action_path, f"action_{self.episode_id:04d}.pkl"), "wb"
        ) as f:
            pickle.dump(self.action_log, f)
        self._saved = True

    def step(self, action, **kwargs):
        obs, reward, done, info = self.env.step(action, **kwargs)
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        self.action_log.append(action)
        return obs, reward, done, info


class CompatibilityWrapper(Wrapper):
    def __init__(self, env):
        super(CompatibilityWrapper, self).__init__(env)

    def step(self, action, **kwargs):
        return convert_to_terminated_truncated_step_api(
            self.env.step(action, **kwargs)
        )


def _extract_craftax_state(state) -> dict:
    """Per-step scalar/array extraction from a single Craftax EnvState.

    Matches the field set saved by dataset_generation/expert_behavior/generate.py
    (modulo per-ep RNG / plants) so the failure analyzer can read playdata and
    eval dumps with the same code path.
    """
    inv = state.inventory
    return {
        "map": np.asarray(state.map, dtype=np.int32),
        "mob_map": np.asarray(state.mob_map, dtype=bool),
        "player_position": np.asarray(state.player_position, dtype=np.int32),
        "player_direction": np.asarray(state.player_direction, dtype=np.int32),
        "player_health": np.asarray(state.player_health, dtype=np.int32),
        "player_food": np.asarray(state.player_food, dtype=np.int32),
        "player_drink": np.asarray(state.player_drink, dtype=np.int32),
        "player_energy": np.asarray(state.player_energy, dtype=np.int32),
        "is_sleeping": np.asarray(state.is_sleeping, dtype=bool),
        "player_recover": np.asarray(state.player_recover, dtype=np.float32),
        "player_hunger": np.asarray(state.player_hunger, dtype=np.float32),
        "player_thirst": np.asarray(state.player_thirst, dtype=np.float32),
        "player_fatigue": np.asarray(state.player_fatigue, dtype=np.float32),
        "light_level": np.asarray(state.light_level, dtype=np.float32),
        "timestep": np.asarray(state.timestep, dtype=np.int32),
        "achievements": np.asarray(state.achievements, dtype=bool),
        "inv_wood": np.asarray(inv.wood, dtype=np.int32),
        "inv_stone": np.asarray(inv.stone, dtype=np.int32),
        "inv_coal": np.asarray(inv.coal, dtype=np.int32),
        "inv_iron": np.asarray(inv.iron, dtype=np.int32),
        "inv_diamond": np.asarray(inv.diamond, dtype=np.int32),
        "inv_sapling": np.asarray(inv.sapling, dtype=np.int32),
        "inv_wood_pickaxe": np.asarray(inv.wood_pickaxe, dtype=np.int32),
        "inv_stone_pickaxe": np.asarray(inv.stone_pickaxe, dtype=np.int32),
        "inv_iron_pickaxe": np.asarray(inv.iron_pickaxe, dtype=np.int32),
        "inv_wood_sword": np.asarray(inv.wood_sword, dtype=np.int32),
        "inv_stone_sword": np.asarray(inv.stone_sword, dtype=np.int32),
        "inv_iron_sword": np.asarray(inv.iron_sword, dtype=np.int32),
        "zombies_position": np.asarray(state.zombies.position, dtype=np.int32),
        "zombies_health": np.asarray(state.zombies.health, dtype=np.int32),
        "zombies_mask": np.asarray(state.zombies.mask, dtype=bool),
        "zombies_attack_cooldown": np.asarray(
            state.zombies.attack_cooldown, dtype=np.int32
        ),
        "skeletons_position": np.asarray(state.skeletons.position, dtype=np.int32),
        "skeletons_health": np.asarray(state.skeletons.health, dtype=np.int32),
        "skeletons_mask": np.asarray(state.skeletons.mask, dtype=bool),
        "skeletons_attack_cooldown": np.asarray(
            state.skeletons.attack_cooldown, dtype=np.int32
        ),
        "arrows_position": np.asarray(state.arrows.position, dtype=np.int32),
        "arrows_mask": np.asarray(state.arrows.mask, dtype=bool),
        "cows_position": np.asarray(state.cows.position, dtype=np.int32),
        "cows_health": np.asarray(state.cows.health, dtype=np.int32),
        "cows_mask": np.asarray(state.cows.mask, dtype=bool),
    }


class StateTrackingWrapper(Wrapper):
    """Dump per-step Craftax EnvState to `{logdir}/state/episode_NNNN.npz`.

    Reads `env.unwrapped._state` after reset/step, accumulates field arrays
    across timesteps, and saves a stacked npz on close.
    """

    def __init__(self, env, logdir: str, episode_id: int | None, **kwargs):
        super(StateTrackingWrapper, self).__init__(env)
        self.episode_id = episode_id
        self.state_path = os.path.join(logdir, "state")
        os.makedirs(self.state_path, exist_ok=True)
        self._buffers: dict = {}
        self._actions: list = []
        self._saved = False

    def begin_episode(self, episode_id: int):
        if self.episode_id is not None and not self._saved:
            self._save()
        _propagate_begin_episode(self.env, episode_id)
        self.episode_id = episode_id
        self._buffers = {}
        self._actions = []
        self._saved = False

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        self._append_current_state()
        return obs, info

    def step(self, action, **kwargs):
        obs, reward, done, info = self.env.step(action, **kwargs)
        if isinstance(action, torch.Tensor):
            a_val = int(action.cpu().numpy())
        elif isinstance(action, str):
            from crafterdojo.env.craftax import _normalize_action
            a_val = _normalize_action(action)
        else:
            a_val = int(np.asarray(action))
        self._actions.append(a_val)
        self._append_current_state()
        return obs, reward, done, info

    def close(self):
        if self.episode_id is not None and not self._saved:
            self._save()
            self._saved = True
        super().close()

    def _current_state(self):
        env = self.env
        while hasattr(env, "env"):
            env = env.env
        return getattr(env, "_state", None)

    def _append_current_state(self):
        state = self._current_state()
        if state is None:
            return
        fields = _extract_craftax_state(state)
        if not self._buffers:
            self._buffers = {k: [v] for k, v in fields.items()}
        else:
            for k, v in fields.items():
                self._buffers[k].append(v)

    def _save(self):
        if not self._buffers:
            return
        stacked = {k: np.stack(v, axis=0) for k, v in self._buffers.items()}
        if self._actions:
            stacked["actions"] = np.asarray(self._actions, dtype=np.int32)
        np.savez_compressed(
            os.path.join(self.state_path, f"episode_{self.episode_id:04d}.npz"),
            **stacked,
        )
        self._saved = True
