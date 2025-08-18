import pathlib
import yaml

root = pathlib.Path(__file__).parent
with open(root / 'data.yaml', 'r') as f:
    for key, value in yaml.safe_load(f).items():
        globals()[key] = value
