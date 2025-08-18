import cv2
import numpy as np
import os
import gymnasium.spaces as spaces
import pickle
import json
from gymnasium import Wrapper

from crafterdojo.env.common.wrappers import BaseEvalTrackingWrapper


class VPTWrapper(Wrapper):
    def __init__(self, env):
        super(VPTWrapper, self).__init__(env)

        obs_space = env.observation_space
        height, width = obs_space.shape[:2]

        self.observation_space = spaces.Dict(
            {"img": spaces.Box(0, 255, shape=(height, width, 3), dtype=np.uint8)}
        )

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
 

class EvalTrackingWrapper(BaseEvalTrackingWrapper):
    def __init__(self, env, logdir: str, episode_id: int, **kwargs):
        super(EvalTrackingWrapper, self).__init__(env, logdir, episode_id, **kwargs)

        video_shape = self.env.observation_space.shape

        video_path = os.path.join(logdir, "video")
        os.makedirs(video_path, exist_ok=True)

        self.info_path = os.path.join(logdir, "info")
        os.makedirs(self.info_path, exist_ok=True)

        self.infos = []

        self.writer = cv2.VideoWriter(
            os.path.join(video_path, f"episode_{episode_id:04d}.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            10,
            (video_shape[1], video_shape[0]),
        )

    def _update_from_obs(self, obs):
        frame = obs.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self.writer.write(frame)

    def _update_from_info(self, info):
        self.infos.append(info.copy())    

    def _save_tracking(self):
        self.writer.release()
        np.savez_compressed(
            os.path.join(self.episode_path, f"episode_{self.episode_id:04d}.npz"),
            **self.log,
        )

        with open(os.path.join(self.info_path, f"episode_{self.episode_id:04d}.pkl"), "wb") as f:
            pickle.dump(self.infos, f)


class AchievementWrapper(Wrapper):
    def __init__(self, env, logdir: str, episode_id: int):
        super(AchievementWrapper, self).__init__(env)

        self.achievement_path = os.path.join(logdir, "achievements")
        os.makedirs(self.achievement_path, exist_ok=True)

        self.achievements = None
        self.episode_id = episode_id

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.achievements = info["achievements"]

        return obs, reward, done, info

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)

        return obs, info

    def close(self):
        self.env.close()

        filepath = os.path.join(self.achievement_path, f"episode_{self.episode_id:04d}.json")
        with open(filepath, "wt") as f:
            json.dump(self.achievements, f)
