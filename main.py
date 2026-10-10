"""Hydra entry point for CrafterDojo FM training and evaluation.

Usage:
    uv run python main.py runner=vpt/bc
    uv run python main.py runner=vpt/bc runner.batch_size=16 wandb.enable=true
    uv run python main.py mode=eval runner=vpt/bc
"""

import functools
import logging

import hydra
from crafterdojo.common.eval_helper import eval_helper
from crafterdojo.common.utils import (
    configure_runtime,
    get_function,
    materialize_runtime_config,
    prepare_fabric,
)
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="main", version_base="1.2")
def main(config: DictConfig):
    materialize_runtime_config(config)
    logdir = HydraConfig.get().runtime.output_dir
    logger.info("Output directory: %s", logdir)
    logger.info("Config:\n%s", OmegaConf.to_yaml(config))

    configure_runtime(config)
    fabric = prepare_fabric(config)
    mode = config.mode

    env_entry = None
    if hasattr(config, "env") and getattr(config.env, "entry", None) is not None:
        env_entry = get_function(config.env.entry)
        env_entry = functools.partial(
            env_entry,
            config=config.env,
            logdir=logdir,
            train=(mode == "train"),
            fabric=fabric,
        )

    if mode == "train":
        train_fn = get_function(config.runner.train_entry)
        train_fn(config=config.runner, logdir=logdir, fabric=fabric, env_entry=env_entry)
    elif mode == "eval":
        runner_entry = get_function(config.runner.eval_entry)
        build_entry_path = getattr(config.runner, "build_entry", None)
        build_entry = get_function(build_entry_path) if build_entry_path else None
        runner_entry = functools.partial(
            eval_helper,
            eval_config=config.eval,
            agent_entry=runner_entry,
            build_entry=build_entry,
        )
        runner_entry(
            config=config.runner, env_entry=env_entry, fabric=fabric, logdir=logdir
        )
    else:
        raise ValueError(f"Unsupported mode: {mode}")


if __name__ == "__main__":
    main()
