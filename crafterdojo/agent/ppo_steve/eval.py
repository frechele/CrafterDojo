import torch
import numpy as np
import os
import pickle
from lightning.fabric import Fabric
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import VecTransposeImage, DummyVecEnv

from crafterdojo.agent.ppo_steve.wrapper import HighLevelWrapper
from crafterdojo.agent.steve1.agent import Steve1Agent
from crafterdojo.model.crafterclip import load_mineclip_wconfig
from crafterdojo.env.common.wrappers import CompatibilityWrapper


@torch.no_grad()
def eval_entry(config, env_entry, episode_id: int, logdir: str, fabric: Fabric):
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

    env = env_entry(episode_id=episode_id)
    env = HighLevelWrapper(
        env,
        config.skillbook,
        steve1_agent,
        config.low_level_steps,
    )
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

    high_level_actions = []

    while not done:
        action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts)
        obs, _, dones, _ = env.step(action)
        episode_starts = dones

        high_level_actions.append(action.item())

        done = dones[0]

    high_level_root = os.path.join(logdir, "hl_action")
    os.makedirs(high_level_root, exist_ok=True)

    with open(os.path.join(high_level_root, f"hl_action_{episode_id:04d}.pkl"), "wb") as f:
        pickle.dump(high_level_actions, f)

    env.close()
