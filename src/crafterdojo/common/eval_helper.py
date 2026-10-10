import json
import os
import multiprocessing as mp
import logging
import numpy as np
from lightning.fabric import Fabric
from crafterdojo.common.crafter_score import compute_crafter_score_from_fraction


def _load_episode_summaries(logdir: str) -> list[dict]:
    summary_dir = os.path.join(logdir, "summary")
    if not os.path.isdir(summary_dir):
        return []

    summaries = []
    for filename in sorted(os.listdir(summary_dir)):
        if not (filename.startswith("episode_") and filename.endswith(".json")):
            continue
        path = os.path.join(summary_dir, filename)
        with open(path, "rt") as f:
            summaries.append(json.load(f))
    return summaries


def _aggregate_episode_summaries(
    episode_summaries: list[dict], requested_episodes: int
) -> dict:
    lengths = [int(summary.get("episode_length", 0)) for summary in episode_summaries]
    rewards = [float(summary.get("total_reward", 0.0)) for summary in episode_summaries]
    success_values = [
        bool(summary["success"])
        for summary in episode_summaries
        if summary.get("success") is not None
    ]
    success_lengths = [
        int(summary.get("episode_length", 0))
        for summary in episode_summaries
        if summary.get("success") is True
    ]
    failure_lengths = [
        int(summary.get("episode_length", 0))
        for summary in episode_summaries
        if summary.get("success") is False
    ]

    achievement_names = sorted(
        {
            name
            for summary in episode_summaries
            for name in summary.get("achievements", {})
        }
    )
    achievement_unlock_rate = {}
    for name in achievement_names:
        achievement_unlock_rate[name] = float(
            np.mean(
                [
                    float(bool(summary.get("achievements", {}).get(name, 0)))
                    for summary in episode_summaries
                ]
            )
        )
    achievement_success_rate_percent = {
        name: rate * 100.0 for name, rate in achievement_unlock_rate.items()
    }

    return {
        "requested_episodes": int(requested_episodes),
        "completed_episodes": int(len(episode_summaries)),
        "completed_episode_ids": [
            int(summary.get("episode_id", -1)) for summary in episode_summaries
        ],
        "mean_episode_length": float(np.mean(lengths)) if lengths else 0.0,
        "mean_success_episode_length": (
            float(np.mean(success_lengths)) if success_lengths else None
        ),
        "median_success_episode_length": (
            float(np.median(success_lengths)) if success_lengths else None
        ),
        "p90_success_episode_length": (
            float(np.percentile(success_lengths, 90)) if success_lengths else None
        ),
        "mean_failure_episode_length": (
            float(np.mean(failure_lengths)) if failure_lengths else None
        ),
        "mean_total_reward": float(np.mean(rewards)) if rewards else 0.0,
        "success_rate": (
            float(np.mean(success_values)) if success_values else None
        ),
        "achievement_unlock_rate": achievement_unlock_rate,
        "achievement_success_rate_percent": achievement_success_rate_percent,
        "crafter_score": compute_crafter_score_from_fraction(achievement_unlock_rate),
    }


def _write_aggregate_summary(logdir: str, eval_config):
    summary_dir = os.path.join(logdir, "summary")
    os.makedirs(summary_dir, exist_ok=True)

    aggregate_path = os.path.join(summary_dir, "aggregate.json")
    episode_summaries = _load_episode_summaries(logdir)
    aggregate = _aggregate_episode_summaries(
        episode_summaries, requested_episodes=eval_config.n_episodes
    )

    with open(aggregate_path, "wt") as f:
        json.dump(aggregate, f, indent=2, sort_keys=True)

    if aggregate["completed_episodes"] != aggregate["requested_episodes"]:
        logging.warning(
            "Evaluation completed %s/%s episodes. See %s for details.",
            aggregate["completed_episodes"],
            aggregate["requested_episodes"],
            aggregate_path,
        )
    else:
        logging.info("Evaluation aggregate summary written to %s", aggregate_path)


def _run_one_episode(
    agent_entry, config, env_entry, episode_id, logdir, fabric, resources
):
    kwargs = dict(
        config=config,
        env_entry=env_entry,
        episode_id=episode_id,
        logdir=logdir,
        fabric=fabric,
    )
    if resources is not None:
        kwargs["_resources"] = resources
    agent_entry(**kwargs)


def _build_resources(build_entry, config, env_entry, fabric):
    if build_entry is None:
        return None
    return build_entry(config=config, env_entry=env_entry, fabric=fabric)


def _close_resources(resources):
    if resources is None:
        return
    env = resources.get("env")
    if env is not None:
        try:
            env.close()
        except Exception as e:
            logging.error(f"Error closing hoisted env: {e}")


def _eval_worker(
    eval_config,
    config,
    episode_ids: list,
    agent_entry,
    env_entry,
    logdir: str,
    fabric: Fabric,
    build_entry=None,
):
    resources = _build_resources(build_entry, config, env_entry, fabric)
    try:
        for episode_id in episode_ids:
            while True:
                try:
                    _run_one_episode(
                        agent_entry, config, env_entry, episode_id, logdir, fabric, resources
                    )
                    break
                except Exception as e:
                    if eval_config.debug:
                        raise e
                    else:
                        logging.error(f"Error in episode {episode_id}: {e}")
    finally:
        _close_resources(resources)


def eval_helper(
    eval_config,
    config,
    agent_entry,
    env_entry,
    logdir: str,
    fabric: Fabric,
    build_entry=None,
):
    start_episode = int(getattr(eval_config, "start_episode", 0))
    if eval_config.n_workers == 1:
        resources = _build_resources(build_entry, config, env_entry, fabric)
        try:
            for episode_id in range(start_episode, start_episode + eval_config.n_episodes):
                while True:
                    try:
                        _run_one_episode(
                            agent_entry, config, env_entry, episode_id, logdir, fabric, resources
                        )
                        break
                    except Exception as e:
                        if eval_config.debug:
                            raise e
                        else:
                            logging.error(f"Error in episode {episode_id}: {e}")
        finally:
            _close_resources(resources)
    else:
        # Static round-robin partition avoids sharing a mp.Queue across spawn
        # workers, which breaks on Python 3.12 where named POSIX semaphores are
        # unlinked eagerly and can't be re-opened in the child.
        ctx = mp.get_context("spawn")
        n_workers = eval_config.n_workers
        partitions = [
            list(range(start_episode + worker_idx, start_episode + eval_config.n_episodes, n_workers))
            for worker_idx in range(n_workers)
        ]

        workers = []
        for worker_idx in range(n_workers):
            worker = ctx.Process(
                target=_eval_worker,
                args=(
                    eval_config,
                    config,
                    partitions[worker_idx],
                    agent_entry,
                    env_entry,
                    logdir,
                    fabric,
                    build_entry,
                ),
            )
            workers.append(worker)
            worker.start()

        for worker in workers:
            worker.join()

    _write_aggregate_summary(logdir, eval_config)
