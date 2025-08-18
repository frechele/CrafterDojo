import os
import json
import pickle
import numpy as np

from lightning.fabric import Fabric

from crafterdojo.lib.VPT.agent import load_model_parameters, CrafterRLAgent


def main(config, env_entry, episode_id: int, logdir: str, fabric: Fabric):
    env = env_entry(episode_id=episode_id)

    device = fabric.device

    agent_policy_kwargs, agent_pi_head_kwargs, lora_kwargs = load_model_parameters(config.model)
    agent = CrafterRLAgent(device, agent_policy_kwargs, agent_pi_head_kwargs, lora_kwargs)

    assert config.weights is not None, "Weights must be provided"
    agent.load_weights(config.weights)
    agent.reset()
    agent.policy.eval()

    obs, info = env.reset()

    done = False
    episode_length = 0
    while not done:
        action = agent.get_action({"pov": obs["img"]})

        obs, _, done, info = env.step(action)

        episode_length += 1

    env.close()
