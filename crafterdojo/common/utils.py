import importlib
from omegaconf import DictConfig, OmegaConf

from lightning.fabric import Fabric


def get_function(target: str):
    module_path, function_name = target.rsplit(".", 1)

    module = importlib.import_module(module_path)
    function = getattr(module, function_name)

    return function


def prepare_fabric(config: DictConfig) -> Fabric:
    logger = None
    if config.mode == "train" and config.wandb.enable:
        from wandb.integration.lightning.fabric import WandbLogger

        logger = WandbLogger(
            project=config.wandb.project,
            group=config.wandb.group,
            name=config.wandb.name,
            config=OmegaConf.to_container(config, resolve=True),
            sync_tensorboard=True,
        )

    fabric = Fabric(
        accelerator=config.fabric.accelerator,
        devices=config.fabric.devices,
        loggers=logger,
    )
    if config.fabric.devices > 1:
        fabric.launch()

    if config.mode == "train":
        fabric.seed_everything(config.seed)

    return fabric


def get_split_range(split_str: str) -> tuple[int, int]:
    split_str = split_str.strip("[]")
    if ":" not in split_str:
        raise ValueError(
            "data_split must be in format '[start_id:]', '[:end_id]', or '[start_id:end_id]'"
        )

    start_str, end_str = split_str.split(":")
    start_id = int(start_str) if start_str else None
    end_id = int(end_str) if end_str else None

    return start_id, end_id
