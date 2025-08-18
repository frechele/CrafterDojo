from gymnasium import Wrapper
import numpy as np
import pickle
import os
import torch
from gymnasium.utils.step_api_compatibility import convert_to_terminated_truncated_step_api


class BaseEvalTrackingWrapper(Wrapper):
    def __init__(self, env, logdir: str, episode_id: int, **kwargs):
        super(BaseEvalTrackingWrapper, self).__init__(env)

        self.obs_log = []
        self.log = dict(
            reward=[],
        )

        self.episode_id = episode_id
        self.episode_path = os.path.join(logdir, "episode")
        os.makedirs(self.episode_path, exist_ok=True)

    def close(self):
        self._save_tracking()
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


class ActionTrackingWrapper(Wrapper):
    def __init__(self, env, logdir: str, episode_id: int, **kwargs):
        super(ActionTrackingWrapper, self).__init__(env)

        self.action_log = []

        self.episode_id = episode_id
        self.action_path = os.path.join(logdir, "action")
        os.makedirs(self.action_path, exist_ok=True)

    def __del__(self):
        with open(
            os.path.join(self.action_path, f"action_{self.episode_id:04d}.pkl"), "wb"
        ) as f:
            pickle.dump(self.action_log, f)

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
        return convert_to_terminated_truncated_step_api(self.env.step(action, **kwargs))
