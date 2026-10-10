import json
from pathlib import Path

import numpy as np

from lightning.fabric import Fabric

from crafterdojo.agent.steve1 import Steve1Agent
from crafterdojo.model.crafterclip import load_mineclip_wconfig, CRAFTER_MINECLIP_CONFIG


def _load_instruction_variants(eval_config) -> list[str] | None:
    variants_path = getattr(eval_config, "instruction_variants_path", None)
    if variants_path is None:
        return None

    path = Path(str(variants_path))
    with path.open("rt") as f:
        payload = json.load(f)

    variants = payload.get("variants")
    if not isinstance(variants, list) or not variants:
        raise ValueError(f"No instruction variants found in {path}")

    n_variants = getattr(eval_config, "num_caption_variants", None)
    if n_variants is not None:
        n_variants = int(n_variants)
        if n_variants <= 0:
            raise ValueError(f"num_caption_variants must be positive, got {n_variants}")
        variants = variants[:n_variants]

    if not variants:
        raise ValueError(f"No instruction variants selected from {path}")
    return [str(v) for v in variants]


def programmatic_eval_text_build(config, env_entry, fabric: Fabric) -> dict:
    device = fabric.device

    clip_config = CRAFTER_MINECLIP_CONFIG.copy()
    clip_config["ckpt"] = config.eval.clip_weights
    mineclip = load_mineclip_wconfig(device, cfg=clip_config)
    mineclip.eval()

    agent = Steve1Agent(
        config.model,
        config.eval.weights,
        config.eval.prior_weights,
        config.eval.cond_scale,
        mineclip,
        device,
        cond_position=getattr(config, "cond_position", "output"),
    )
    instruction_variants = _load_instruction_variants(config.eval)
    agent.set_goal(
        instruction_variants[0] if instruction_variants else config.eval.instruction
    )

    env = env_entry(episode_id=None)

    return {
        "mineclip": mineclip,
        "agent": agent,
        "env": env,
        "instruction_variants": instruction_variants,
    }


def programmatic_eval_text(
    config,
    env_entry,
    episode_id: int,
    logdir: str,
    fabric: Fabric,
    _resources: dict | None = None,
):
    own_resources = _resources is None
    if own_resources:
        _resources = programmatic_eval_text_build(config, env_entry, fabric)

    agent = _resources["agent"]
    env = _resources["env"]
    instruction_variants = _resources.get("instruction_variants")

    env.begin_episode(episode_id)
    if instruction_variants:
        agent.set_goal(instruction_variants[episode_id % len(instruction_variants)])
    agent.reset()
    obs, info = env.reset(seed=episode_id)

    done = False
    while not done:
        action = agent.get_action(obs)
        obs, _, done, info = env.step(action)

    if own_resources:
        env.close()


def programmatic_eval_video(
    config, env_entry, episode_id: int, logdir: str, fabric: Fabric
):
    device = fabric.device

    clip_config = CRAFTER_MINECLIP_CONFIG.copy()
    clip_config["ckpt"] = config.eval.clip_weights
    mineclip = load_mineclip_wconfig(device, cfg=clip_config)
    mineclip.eval()

    agent = Steve1Agent(
        config.model,
        config.eval.weights,
        config.eval.prior_weights,
        config.eval.cond_scale,
        mineclip,
        device,
        cond_position=getattr(config, "cond_position", "output"),
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
