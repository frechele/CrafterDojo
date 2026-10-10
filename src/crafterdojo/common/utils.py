import importlib
import os
import sys
from typing import TYPE_CHECKING

from omegaconf import DictConfig, OmegaConf

if TYPE_CHECKING:
    from lightning.fabric import Fabric


_WANDB_PROJECT_BY_AGENT = {
    "vpt_bc": "crafterdojo-vpt",
    "steve1_bc": "crafterdojo-steve1",
    "crafterclip": "crafterdojo-crafterclip",
}


def get_function(target: str):
    module_path, function_name = target.rsplit(".", 1)

    module = importlib.import_module(module_path)
    function = getattr(module, function_name)

    return function


def configure_runtime(config: DictConfig):
    if config.mode != "eval":
        return

    eval_config = getattr(config, "eval", None)
    requested_platform = getattr(eval_config, "jax_platform", None)
    if requested_platform is None:
        return

    requested_platform = str(requested_platform).strip().lower()
    if requested_platform in {"", "none", "null"}:
        return

    if "jax" in sys.modules:
        raise RuntimeError(
            "JAX was imported before eval runtime configuration. "
            "Set eval.jax_platform before importing Craftax/JAX modules."
        )

    os.environ["JAX_PLATFORMS"] = requested_platform
    os.environ["JAX_PLATFORM_NAME"] = requested_platform


def resolve_wandb_project(config: DictConfig) -> str:
    wandb_config = getattr(config, "wandb", None)
    project = getattr(wandb_config, "project", None)

    if project is not None:
        project = str(project).strip()
    if project and project.lower() != "auto":
        return project

    agent_name = str(getattr(getattr(config, "agent", None), "name", "")).strip().lower()
    if agent_name in _WANDB_PROJECT_BY_AGENT:
        return _WANDB_PROJECT_BY_AGENT[agent_name]
    if agent_name:
        return f"crafterdojo-{agent_name.replace('_', '-')}"

    return "crafterdojo"


def materialize_runtime_config(config: DictConfig):
    wandb_config = getattr(config, "wandb", None)
    if wandb_config is not None and hasattr(wandb_config, "project"):
        wandb_config.project = resolve_wandb_project(config)


def prepare_fabric(config: DictConfig):
    from lightning.fabric import Fabric
    from lightning.fabric.strategies import DDPStrategy
    import torch

    matmul_precision = getattr(config.fabric, "matmul_precision", None)
    if matmul_precision:
        torch.set_float32_matmul_precision(str(matmul_precision))

    logger = None
    if config.mode == "train" and config.wandb.enable:
        from wandb.integration.lightning.fabric import WandbLogger

        logger = WandbLogger(
            project=resolve_wandb_project(config),
            group=config.wandb.group,
            name=config.wandb.name,
            config=OmegaConf.to_container(config, resolve=True),
            mode=config.wandb.mode,
            sync_tensorboard=False,
        )

    precision = getattr(config.fabric, "precision", None)

    if config.fabric.devices > 1:
        find_unused_parameters = bool(
            getattr(config.fabric, "find_unused_parameters", False)
        )
        strategy = DDPStrategy(find_unused_parameters=find_unused_parameters)

        fabric = Fabric(
            accelerator=config.fabric.accelerator,
            devices=config.fabric.devices,
            precision=precision,
            loggers=logger,
            strategy=strategy,
        )
    else:
        fabric = Fabric(
            accelerator=config.fabric.accelerator,
            devices=config.fabric.devices,
            precision=precision,
            loggers=logger,
        )

    if config.fabric.devices > 1:
        fabric.launch()

    if config.mode == "train":
        fabric.seed_everything(config.seed)

    return fabric


def get_split_range(split_str: str) -> list[tuple[int | None, int | None]]:
    ranges = []
    for raw_range in split_str.split("+"):
        range_str = raw_range.strip()
        if not range_str:
            continue

        if not (range_str.startswith("[") and range_str.endswith("]")):
            raise ValueError(
                "data_split must be surrounded by brackets: e.g. '[start_id:end_id]'"
            )

        inner = range_str.strip("[]")
        if ":" not in inner:
            raise ValueError(
                "data_split must contain ':' in each range, e.g. '[start_id:]', '[:end_id]', '[start_id:end_id]'"
            )

        start_str, end_str = inner.split(":", 1)
        start_id = int(start_str) if start_str else None
        end_id = int(end_str) if end_str else None

        ranges.append((start_id, end_id))

    if not ranges:
        raise ValueError(
            "data_split must include at least one range like '[start_id:end_id]' and can be combined with '+'"
        )

    return ranges
