import pathlib
import glob

import yaml


task_root = pathlib.Path(__file__).parent / "tasks"
task_files = glob.glob(str(task_root / "*.yaml"))
TASKS = {}
for task_file in task_files:
    with open(task_file, "rt") as f:
        local_tasks = yaml.safe_load(f)
        if local_tasks is None:
            continue

        TASKS.update(
            {f"{task['task_type']}_{k}": task for k, task in local_tasks.items()}
        )
