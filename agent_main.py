import hydra
import logging
import functools
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from crafterdojo.common.eval_helper import eval_helper
from crafterdojo.common.utils import get_function, prepare_fabric


@hydra.main(config_path="config", config_name="agent_main", version_base="1.2")
def main(config: DictConfig):
    logdir = HydraConfig.get().runtime.output_dir
    logging.info(f"Output Directory: {logdir}")

    fabric = prepare_fabric(config)

    mode = config.mode
    if mode == "train":
        train = True
        agent_entry = get_function(config.agent.train_entry)
    elif mode == "eval":
        train = False
        agent_entry = get_function(config.agent.eval_entry)
        agent_entry = functools.partial(
            eval_helper, eval_config=config.eval, agent_entry=agent_entry
        )
    else:
        raise ValueError(f"Invalid mode: {mode}")

    env_entry = None
    if config.env.entry is not None:
        env_entry = get_function(config.env.entry)
        env_entry = functools.partial(
            env_entry, config=config.env, logdir=logdir, train=train, fabric=fabric
        )

    agent_entry(config=config.agent, env_entry=env_entry, fabric=fabric, logdir=logdir)


if __name__ == "__main__":
    main()
