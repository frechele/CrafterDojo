"""DEPS / H-DEP evaluation entry compatible with `crafterdojo.common.eval_helper`."""

import json
import logging
import os

import torch
from lightning.fabric import Fabric
from omegaconf import OmegaConf

from crafterdojo.agent.deps.agent import DEPSAgent, DEPSObservation
from crafterdojo.agent.deps.controller import DEPSController
from crafterdojo.agent.deps.knowledge_graph import (
    KnowledgeGraphPlanner,
    StagedKnowledgeGraphPlanner,
)
from crafterdojo.agent.deps.planner import PlannerFailure
from crafterdojo.agent.steve1 import SBERTSteve1Agent, Steve1Agent
from crafterdojo.model.crafterclip import (
    CRAFTER_MINECLIP_CONFIG,
    load_mineclip_wconfig,
)


def _build_steve1(config, device: torch.device):
    steve1_cfg = config.steve1
    steve1_type = getattr(steve1_cfg, "type", "mineclip")

    if steve1_type == "sbert":
        return SBERTSteve1Agent(
            steve1_cfg.model,
            steve1_cfg.weights,
            steve1_cfg.cond_scale,
            steve1_cfg.sbert_model_name,
            device,
            goal_embed_dim=getattr(steve1_cfg, "goal_embed_dim", 768),
            cond_position=getattr(steve1_cfg, "cond_position", "output"),
        )

    clip_weights = getattr(steve1_cfg, "clip_weights", None)
    if clip_weights:
        clip_cfg = CRAFTER_MINECLIP_CONFIG.copy()
        clip_cfg["ckpt"] = clip_weights
    else:
        clip_cfg = None
    mineclip = load_mineclip_wconfig(device, cfg=clip_cfg)
    mineclip.eval()

    return Steve1Agent(
        steve1_cfg.model,
        steve1_cfg.weights,
        steve1_cfg.prior_weights,
        steve1_cfg.cond_scale,
        mineclip,
        device,
        cond_position=getattr(steve1_cfg, "cond_position", "output"),
    )


def _build_planner(config):
    planner_backend = getattr(config, "planner_backend", "knowledge_graph")
    if planner_backend == "knowledge_graph":
        return KnowledgeGraphPlanner()

    if planner_backend in ("knowledge_graph_staged", "hdepv2"):
        return StagedKnowledgeGraphPlanner()

    if planner_backend == "llm":
        from crafterdojo.agent.deps.planner import DEPSPlanner

        planner_cfg = OmegaConf.to_container(config.planner, resolve=True) or {}
        model_name = planner_cfg.pop("model_name")
        embed_crafter = planner_cfg.pop("embed_crafter", False)
        temperature = planner_cfg.pop("temperature", 0.0)
        return DEPSPlanner(
            model_name=model_name,
            temperature=temperature,
            embed_crafter=embed_crafter,
            **planner_cfg,
        )

    raise ValueError(f"Unknown planner_backend: {planner_backend}")


def _resolve_target_item(config, env, info: dict | None) -> str:
    target_item = None
    unwrapped = getattr(env, "unwrapped", env)
    target_item = getattr(unwrapped, "target_item", None)
    if target_item is None and info is not None:
        target_item = info.get("target_item")
    if target_item is None:
        target_item = getattr(config, "target_item", None)
    if target_item is None:
        raise ValueError(
            "DEPS requires a target_item: set env task with target_item or runner.target_item."
        )
    return str(target_item)


def _configure_logger(logdir: str, episode_id: int) -> logging.Logger:
    deps_log_dir = os.path.join(logdir, "deps_logs")
    os.makedirs(deps_log_dir, exist_ok=True)

    logger = logging.getLogger("deps")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)

    handler = logging.FileHandler(
        os.path.join(deps_log_dir, f"episode_{episode_id:04d}.log")
    )
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    return logger


@torch.no_grad()
def eval_build(config, env_entry, fabric: Fabric) -> dict:
    """Hoisted resources: Steve-1 controller + planner + env (reused across episodes)."""
    device = fabric.device

    steve1 = _build_steve1(config, device)
    planner = _build_planner(config)

    controller = DEPSController(steve1=steve1)

    survival = OmegaConf.to_container(config.survival, resolve=True) if hasattr(config, "survival") else None
    agent = DEPSAgent(planner=planner, controller=controller, survival=survival)

    env = env_entry(episode_id=None)

    return {
        "steve1": steve1,
        "planner": planner,
        "controller": controller,
        "agent": agent,
        "env": env,
    }


@torch.no_grad()
def eval_entry(
    config,
    env_entry,
    episode_id: int,
    logdir: str,
    fabric: Fabric,
    _resources: dict | None = None,
):
    own_resources = _resources is None
    if own_resources:
        _resources = eval_build(config, env_entry, fabric)

    agent: DEPSAgent = _resources["agent"]
    env = _resources["env"]
    planner = _resources["planner"]

    _configure_logger(logdir, episode_id)

    env.begin_episode(episode_id)
    obs, info = env.reset(seed=episode_id)

    target_item = _resolve_target_item(config, env, info)
    deps_obs = DEPSObservation(obs, target_item)

    planner_failure_reason: str | None = None
    planner_failure_response: str | None = None
    initial_plan: list = []

    try:
        agent.reset(deps_obs, info)
        initial_plan = [plan_text for _, plan_text in planner.plans]
    except PlannerFailure as e:
        planner_failure_reason = e.reason
        planner_failure_response = e.raw_response
        logging.getLogger("deps").warning(
            "Episode %s aborted: planner failed at initial plan (%s)",
            episode_id,
            e.reason,
        )

    max_steps = getattr(config, "max_steps", None)

    done = False
    step = 0
    if planner_failure_reason is None:
        while not done and (max_steps is None or step < max_steps):
            try:
                action = agent.get_action(deps_obs, info)
            except PlannerFailure as e:
                planner_failure_reason = e.reason
                planner_failure_response = e.raw_response
                logging.getLogger("deps").warning(
                    "Episode %s aborted at step %s: planner failure (%s)",
                    episode_id,
                    step,
                    e.reason,
                )
                break
            obs, _, done, info = env.step(action)
            deps_obs = DEPSObservation(obs, target_item)
            step += 1

    timed_out = max_steps is not None and step >= max_steps and not done
    final_info = info or {}

    deps_summary_dir = os.path.join(logdir, "deps_summary")
    os.makedirs(deps_summary_dir, exist_ok=True)
    deps_summary = {
        "episode_id": int(episode_id),
        "target_item": target_item,
        "success": bool(final_info.get("is_success", final_info.get("success", False))),
        "timed_out": bool(timed_out),
        "steps": int(step),
        "max_steps": (int(max_steps) if max_steps is not None else None),
        "final_inventory": _json_inventory(final_info.get("inventory", {})),
        "initial_plan": initial_plan,
        "remaining_plan": [plan_text for _, plan_text in planner.plans],
        "token_usage": {
            k: int(v) for k, v in getattr(planner, "tokens_counter", {}).items()
        },
        "planner_failed": planner_failure_reason is not None,
        "planner_failure_reason": planner_failure_reason,
        "planner_failure_response": planner_failure_response,
    }
    if planner_failure_reason is not None:
        deps_summary["success"] = None
        # force env auto-save to record success=None so this episode is
        # excluded from SR (aggregator skips summaries with success is None).
        target = env
        for _ in range(10):
            if target is None:
                break
            if hasattr(target, "last_info"):
                target.last_info = None
                break
            target = getattr(target, "env", None)
    with open(
        os.path.join(deps_summary_dir, f"episode_{episode_id:04d}.json"), "wt"
    ) as f:
        json.dump(deps_summary, f, indent=2)

    if own_resources:
        env.close()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _json_inventory(inventory: dict) -> dict:
    out = {}
    for k, v in (inventory or {}).items():
        try:
            out[str(k)] = int(v)
        except (TypeError, ValueError):
            out[str(k)] = v
    return out
