import pathlib
import yaml
import glob

from .meta.achievement_meta import AchievementMeta
from .meta.conditional_meta import ConditionalAchievementMeta, ConditionalAchievement3Meta
from .meta.harvest_meta import HarvestMeta


task_root = pathlib.Path(__file__).parent / "tasks"
task_files = glob.glob(str(task_root / "*.yaml"))
TASKS = {}
for task_file in task_files:
    with open(task_file, "rt") as f:
        local_tasks = yaml.safe_load(f)
        if local_tasks is None:
            continue

        TASKS.update({
            f"{task['task_type']}_{k}": task
            for k, task in local_tasks.items()
        })


META_CLASSES = {
    "achievement": AchievementMeta,

    "conditional": ConditionalAchievementMeta,
    "conditional_3": ConditionalAchievement3Meta,

    "harvest": HarvestMeta,
}

def make_env(task_name: str, seed=None):
    task = TASKS[task_name].copy()
    task_type = task.pop("task_type")
    meta_cls = META_CLASSES[task_type]

    return meta_cls(**task, seed=seed)
