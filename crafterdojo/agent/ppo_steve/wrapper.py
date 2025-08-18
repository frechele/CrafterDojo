import numpy as np
import torch
import gymnasium as gym
import gymnasium.spaces as spaces
import pickle

from crafterdojo.agent.steve1.agent import Steve1Agent


class HighLevelWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        skillbook_path: str,
        steve1_agent: Steve1Agent,
        low_level_steps: int,
    ):
        super(HighLevelWrapper, self).__init__(env)
        self.steve1_agent = steve1_agent
        self.low_level_steps = low_level_steps

        with open(skillbook_path, "rb") as f:
            skillbook = pickle.load(f)
            assert isinstance(skillbook, list), "Skillbook must be an ordered list"
            self.skillbook = sorted(skillbook)

    @property
    def observation_space(self):
        return self.env.observation_space["img"]

    @property
    def action_space(self):
        return spaces.Discrete(len(self.skillbook))

    def reset(self, *args, **kwargs):
        self.steve1_agent.reset()
        self.total_steps = 0

        obs, info = super(HighLevelWrapper, self).reset(*args, **kwargs)
        self._last_obs = obs["img"]

        return obs["img"], info

    def step(self, action):
        skill = self.skillbook[action]
        self.steve1_agent.set_goal(skill)

        cum_reward = 0
        actual_steps = 0
        for _ in range(self.low_level_steps):
            with torch.no_grad():
                action = self.steve1_agent.get_action({"img": self._last_obs})
            obs, reward, done, info = self.env.step(action)
            cum_reward += reward

            self.total_steps += 1
            self._last_obs = obs["img"]

            actual_steps += 1

            if done:
                break

        info["actual_steps"] = actual_steps

        return self._last_obs, cum_reward, done, info


class HighLevelVPTWrapper(HighLevelWrapper):
    @property
    def observation_space(self):
        return self.env.observation_space

    def reset(self, *args, **kwargs):
        obs, info = super().reset(*args, **kwargs)
        return {"img": obs}, info

    def step(self, action):
        obs, reward, done, info = super().step(action)
        return {"img": obs}, reward, done, info
