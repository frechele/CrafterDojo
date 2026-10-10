"""Crafter-Classic constants table used by the DEPS knowledge graph and controller.

The DEPS reference implementation reads `crafter.constants` (loaded from the
upstream Crafter data.yaml). Craftax-Classic does not expose the same dict
shape, so we vendor the YAML and re-export it under stable names: `actions`,
`materials`, `walkable`, `items`, `collect`, `place`, `make`, `achievements`.
"""

import pathlib

import yaml


_DATA_PATH = pathlib.Path(__file__).with_name("crafter_classic_data.yaml")
with _DATA_PATH.open("rt") as _f:
    for _key, _value in yaml.safe_load(_f).items():
        globals()[_key] = _value
