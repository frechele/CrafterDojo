import os
import torch
import numpy as np
from lightning.fabric import Fabric
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.vec_env.vec_transpose import VecTransposeImage
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from crafterdojo.agent.vpt.logging import LoggingCallback
from crafterdojo.env.common.wrappers import CompatibilityWrapper
from crafterdojo.env.crafter.wrappers import DeathPenaltyWrapper
from crafterdojo.lib.VPT.agent import load_model_parameters
from crafterdojo.lib.VPTPPO.algorithm import VPTPPO
from crafterdojo.lib.VPTPPO.policy import VPTPolicy


class CustomEvalCallback(EvalCallback):
    def _on_training_start(self):
        super()._on_training_start()

        self._is_success_buffer = []

        episode_rewards, episode_lengths = evaluate_policy(
            self.model,
            self.eval_env,
            n_eval_episodes=self.n_eval_episodes,
            render=self.render,
            deterministic=self.deterministic,
            return_episode_rewards=True,
            warn=self.warn,
            callback=self._log_success_callback,
        )

        mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
        mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
        self.last_mean_reward = float(mean_reward)

        if self.verbose >= 1:
            print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
            print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
        # Add to current Logger
        self.logger.record("eval/mean_reward", float(mean_reward))
        self.logger.record("eval/mean_ep_length", mean_ep_length)

        if len(self._is_success_buffer) > 0:
            success_rate = np.mean(self._is_success_buffer)
            if self.verbose >= 1:
                print(f"Success rate: {100 * success_rate:.2f}%")
            self.logger.record("eval/success_rate", success_rate)

        self.logger.dump(self.num_timesteps)

def main(config, env_entry, logdir: str, fabric: Fabric):
    checkpoint_path = os.path.join(logdir, "checkpoints")
    os.makedirs(checkpoint_path, exist_ok=True)
    tensorboard_path = os.path.join(logdir, "tensorboard")

    device = fabric.device
    
    def _make_env():
        env = env_entry()
        env = DeathPenaltyWrapper(env, config.live_bonus, config.death_penalty)
        env = CompatibilityWrapper(env)
        return env

    if config.n_envs > 1:
        vec_cls = SubprocVecEnv
    else:
        vec_cls = DummyVecEnv

    env = vec_cls([_make_env for _ in range(config.n_envs)])
    env = VecTransposeImage(env, skip=True)

    policy_kwargs, pi_head_kwargs, lora_kwargs = load_model_parameters(config.model)

    model = VPTPPO(
        VPTPolicy,
        env,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        kl_coef=config.kl_coef,
        kl_decay=config.kl_decay,
        verbose=1,
        tensorboard_log=tensorboard_path,
        device=device,
        policy_kwargs=dict(
            config=config,
            policy_kwargs=policy_kwargs,
            pi_head_kwargs=pi_head_kwargs,
            weights_path=config.weights,
            lora_kwargs=lora_kwargs,
        ),
    )

    eval_env = vec_cls([lambda: Monitor(_make_env())])
    eval_env = VecTransposeImage(eval_env, skip=True)
    
    model.learn(
        total_timesteps=config.total_timesteps,
        callback=[
            LoggingCallback(),
            CheckpointCallback(save_freq=config.save_freq, save_path=checkpoint_path),
            CustomEvalCallback(eval_env, best_model_save_path=checkpoint_path, 
                        log_path=tensorboard_path, eval_freq=config.eval_freq,
                        n_eval_episodes=config.n_eval_episodes),
        ],
    )

    # save last checkpoint
    torch.save(
        model.policy.model.state_dict(), os.path.join(checkpoint_path, "last.pth")
    )
