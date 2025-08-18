import os
from sb3_contrib import RecurrentPPO
from lightning.fabric import Fabric
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.vec_env.vec_transpose import VecTransposeImage
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from crafterdojo.agent.vpt.rl_train import CustomEvalCallback
from crafterdojo.agent.ppo_steve.wrapper import HighLevelWrapper
from crafterdojo.agent.ppo_steve.logging import LoggingCallback
from crafterdojo.agent.steve1.agent import Steve1Agent
from crafterdojo.env.common.wrappers import CompatibilityWrapper
from crafterdojo.model.crafterclip import load_mineclip_wconfig


def train_entry(config, env_entry, logdir: str, fabric: Fabric):
    checkpoint_path = os.path.join(logdir, "checkpoints")
    os.makedirs(checkpoint_path, exist_ok=True)
    tensorboard_path = os.path.join(logdir, "tensorboard")

    device = fabric.device

    mineclip = load_mineclip_wconfig(device)
    mineclip.eval()

    steve1_agent = Steve1Agent(
        config.steve1.model,
        config.steve1.weights,
        config.steve1.prior_weights,
        config.steve1.cond_scale,
        mineclip,
        device,
    )

    def _make_env():
        env = env_entry()
        env = HighLevelWrapper(
            env,
            config.skillbook,
            steve1_agent,
            config.low_level_steps,
        )
        env = CompatibilityWrapper(env)
        return env

    if config.n_envs > 1:
        env = SubprocVecEnv([_make_env for _ in range(config.n_envs)])
    else:
        env = DummyVecEnv([_make_env])
    env = VecTransposeImage(env)

    model = RecurrentPPO(
        "CnnLstmPolicy",
        env,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        verbose=1,
        tensorboard_log=tensorboard_path,
        device=device,
    )

    eval_env = DummyVecEnv([lambda: Monitor(_make_env())])
    eval_env = VecTransposeImage(eval_env)
    
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
