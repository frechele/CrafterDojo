import numpy as np
from datetime import datetime
from stable_baselines3.common.callbacks import BaseCallback


class LoggingCallback(BaseCallback):
    def __init__(self):
        super(LoggingCallback, self).__init__()

        self.iter_rewards = []
        self.num_dones = 0
        self.num_timesteps = 0
        self.rollout_steps = 0

        self.iteration = 0

        self.num_iterations = 0

    def _on_training_start(self):
        self.cum_rewards = np.zeros(self.training_env.num_envs)
        self.episode_steps = np.zeros(self.training_env.num_envs)
        self.selected_skills = [[] for _ in range(self.training_env.num_envs)]

        self.num_timesteps = self.num_iterations

    def _on_step(self):
        self.cum_rewards += self.locals["rewards"]
        
        actual_steps = np.zeros(self.training_env.num_envs)
        for env_id in range(self.training_env.num_envs):
            infos = self.locals["infos"][env_id]
            actual_steps[env_id] = infos["actual_steps"]

        self.episode_steps += actual_steps
        
        eps_rewards = self.cum_rewards[np.where(self.locals["dones"])].tolist()
        eps_lengths = self.episode_steps[np.where(self.locals["dones"])].tolist()
        
        for r in eps_rewards:
            self.logger.record_mean("custom/reward", r)
        for length in eps_lengths:
            self.logger.record_mean("custom/episode_length", length)
            
        self.iter_rewards += eps_rewards
        self.iter_episode_lengths += eps_lengths
        
        self.cum_rewards *= (1 - self.locals["dones"])
        self.episode_steps *= (1 - self.locals["dones"])

        for i, done in enumerate(self.locals["dones"]):
            if done:
                self.selected_skills[i] = []

        for i, skill in enumerate(self.locals["actions"]):
            self.selected_skills[i].append(skill)

        self.num_dones += np.sum(self.locals["dones"])
        self.rollout_steps += self.training_env.num_envs

        for env_id in range(self.training_env.num_envs):
            infos = self.locals["infos"][env_id]
            self.num_iterations += infos["actual_steps"]

            if "is_success" in infos:
                self.iter_successes.append(infos["is_success"] == True)

            if not self.locals["dones"][env_id]:
                continue

            if "achievements" in infos:
                num_achievements = sum([v > 0 for v in infos["achievements"].values()])
                self.iter_achievements.append(num_achievements)

        self.model.num_timesteps = self.num_iterations

        return True

    def _on_rollout_start(self):
        self.num_dones = 0
        self.rollout_steps = 0
        
        self.rollout_start_time = datetime.now()
        self.iter_rewards = []
        self.iter_achievements = []
        self.iter_successes = []
        self.iter_episode_lengths = []

        print("Starting rollout")


    def _on_rollout_end(self):
        self.update_start_time = datetime.now()
        print("Finished rollout in", self.update_start_time - self.rollout_start_time)
        print("\tMax reward:", np.amax(self.iter_rewards) if self.num_dones > 0 else 0)
        print("\tLast rewards:", np.mean(self.iter_rewards))
        print("\tNum dones:", self.num_dones)
        
        if len(self.iter_episode_lengths) > 0:
            print("\tMean episode length:", np.mean(self.iter_episode_lengths))
            print("\tMax episode length:", np.amax(self.iter_episode_lengths))

        num_skills = np.ones(self.training_env.num_envs)
        for i, skills in enumerate(self.selected_skills):
            prev_skill = skills[0]
            for skill in skills[1:]:
                if skill != prev_skill:
                    num_skills[i] += 1
                    prev_skill = skill

        self.logger.record("custom/rollout_time", (self.update_start_time - self.rollout_start_time).total_seconds())
        self.logger.record("custom/completed_episodes", self.num_dones)
        self.logger.record("custom/max_reward", np.amax(self.iter_rewards) if self.num_dones > 0 else 0)
        self.logger.record("custom/mean_reward", np.mean(self.iter_rewards))
        self.logger.record("custom/num_skills", np.mean(num_skills))
        self.logger.record("custom/num_unique_skills", np.mean([len(set(s)) for s in self.selected_skills]))

        if len(self.iter_achievements) > 0:
            self.logger.record("custom/mean_achievements", np.mean(self.iter_achievements))
        if len(self.iter_successes) > 0:
            self.logger.record("custom/mean_successes", np.mean(self.iter_successes))
        if len(self.iter_episode_lengths) > 0:
            self.logger.record("custom/mean_episode_length", np.mean(self.iter_episode_lengths))
            self.logger.record("custom/max_episode_length", np.amax(self.iter_episode_lengths))

        print("Starting update")
