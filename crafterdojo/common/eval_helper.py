import multiprocessing as mp
import numpy as np
import logging
from lightning.fabric import Fabric


def _eval_worker(
    eval_config,
    config,
    id_cluster: np.array,
    agent_entry,
    env_entry,
    logdir: str,
    fabric: Fabric,
):
    for episode_id in id_cluster:
        while True:
            try:
                agent_entry(
                    config=config,
                    env_entry=env_entry,
                    episode_id=episode_id,
                    logdir=logdir,
                    fabric=fabric,
                )
                break
            except Exception as e:
                if eval_config.debug:
                    raise e
                else:
                    logging.error(f"Error in episode {episode_id}: {e}")


def eval_helper(
    eval_config, config, agent_entry, env_entry, logdir: str, fabric: Fabric
):
    if eval_config.n_workers == 1:
        cluster = np.arange(eval_config.n_episodes)
        _eval_worker(
            eval_config, config, cluster, agent_entry, env_entry, logdir, fabric
        )
    else:
        mp.set_start_method("spawn")

        episode_ids = np.arange(eval_config.n_episodes)
        id_clusters = np.array_split(episode_ids, eval_config.n_workers)

        workers = []
        for id_cluster in id_clusters:
            worker = mp.Process(
                target=_eval_worker,
                args=(
                    eval_config,
                    config,
                    id_cluster,
                    agent_entry,
                    env_entry,
                    logdir,
                    fabric,
                ),
            )
            workers.append(worker)
            worker.start()

        for worker in workers:
            worker.join()
