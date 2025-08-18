import hydra
import logging
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from crafterdojo.common.utils import get_function, prepare_fabric


@hydra.main(config_path="config", config_name="model_main", version_base="1.2")
def main(config: DictConfig):
    logdir = HydraConfig.get().runtime.output_dir
    logging.info(f"Output Directory: {logdir}")

    fabric = prepare_fabric(config)

    mode = config.mode
    if mode == "train":
        model_entry = get_function(config.model.train_entry)
    elif mode == "eval":
        model_entry = get_function(config.model.eval_entry)
    else:
        raise ValueError(f"Invalid mode: {mode}")

    model_entry(config=config.model, fabric=fabric, logdir=logdir)


if __name__ == "__main__":
    main()
