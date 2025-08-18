import torch
import numpy as np
from lightning.fabric import Fabric
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import VecTransposeImage, DummyVecEnv

from crafterdojo.env.common.wrappers import CompatibilityWrapper


@torch.no_grad()
def eval_entry(config, env_entry, episode_id: int, logdir: str, fabric: Fabric):
    device = fabric.device

    env = env_entry(episode_id=episode_id).env
    env = CompatibilityWrapper(env)
    env = DummyVecEnv([lambda: env])
    env = VecTransposeImage(env)

    assert config.eval.load is not None
    
    model = RecurrentPPO.load(config.eval.load)
    model.policy.set_training_mode(False)

    done = False
    obs = env.reset()

    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)

    while not done:
        action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts)
        obs, _, dones, _ = env.step(action)
        episode_starts = dones

        done = dones[0]

    env.close()
