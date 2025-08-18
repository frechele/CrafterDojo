from . import crafter
from typing import Optional

from lightning.fabric import Fabric

from crafterdojo.env.common.wrappers import ActionTrackingWrapper
from crafterdojo.env.crafter.wrappers import EvalTrackingWrapper, VPTWrapper, AchievementWrapper
from crafterdojo.env.crafter.crafterdojo import make_env


def crafter_main(
    config, logdir: str, train: bool, fabric: Fabric, episode_id: Optional[int] = None
):
    env = crafter.Env(size=(144, 144), seed=episode_id)

    if not train:
        env = EvalTrackingWrapper(env, logdir, episode_id)
        env = ActionTrackingWrapper(env, logdir, episode_id)
        env = AchievementWrapper(env, logdir, episode_id)

    env = VPTWrapper(env)
    return env


def crafterdojo_main(
    config, logdir: str, train: bool, fabric: Fabric, episode_id: Optional[int] = None
):
    task = config["task"]
    assert task is not None

    env = make_env(task, seed=episode_id)

    if not train:
        env = EvalTrackingWrapper(env, logdir, episode_id)
        env = ActionTrackingWrapper(env, logdir, episode_id)

    env = VPTWrapper(env)
    return env
