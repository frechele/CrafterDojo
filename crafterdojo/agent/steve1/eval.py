import os
import json
import pickle
import numpy as np

from lightning.fabric import Fabric

from crafterdojo.agent.steve1.agent import Steve1Agent
from crafterdojo.model.crafterclip import load_mineclip_wconfig, CRAFTER_MINECLIP_CONFIG


def programmatic_eval_text(config, env_entry, episode_id: int, logdir: str, fabric: Fabric):
    device = fabric.device

    clip_config = CRAFTER_MINECLIP_CONFIG.copy()
    clip_config["ckpt"] = config.eval.clip_weights
    mineclip = load_mineclip_wconfig(device, cfg=clip_config)
    mineclip.eval()

    agent = Steve1Agent(
        config.model,
        config.weights,
        config.eval.prior_weights,
        config.eval.cond_scale,
        mineclip,
        device,
    )

    env = env_entry(episode_id=episode_id)

    agent.reset()
    agent.set_goal(config.eval.instruction)

    obs, info = env.reset()

    done = False
    while not done:
        action = agent.get_action(obs)
        obs, _, done, info = env.step(action)

    env.close()


def programmatic_eval_video(config, env_entry, episode_id: int, logdir: str, fabric: Fabric):
    device = fabric.device

    clip_config = CRAFTER_MINECLIP_CONFIG.copy()
    clip_config["ckpt"] = config.eval.clip_weights
    mineclip = load_mineclip_wconfig(device, cfg=clip_config)
    mineclip.eval()

    agent = Steve1Agent(
        config.model,
        config.weights,
        config.eval.prior_weights,
        config.eval.cond_scale,
        mineclip,
        device,
    )

    env = env_entry(episode_id=episode_id)

    video_embeddings = np.load(config.eval.video_embeddings)    

    agent.reset()
    embedding = video_embeddings[config.eval.instruction]
    agent.set_goal(embedding)

    obs, info = env.reset()

    done = False
    while not done:
        action = agent.get_action(obs)
        obs, _, done, info = env.step(action)

    env.close()
