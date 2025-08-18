# CrafterDojo Caption Generator Toolkit

The Caption Generator toolkit uses a rule-based caption generation system that monitors agent behavior and automatically creates descriptive captions for corresponding video segments.

## Quick Start

To generate the CrafterCaption dataset, execute the following command:

```bash
uv run bash scripts/run.sh
```


## Customize Rule Set

The rule set should be defined in the `toolkit/caption/src/rule` directory.

```
toolkit/caption/src/rule/
├── __init__.py
├── achievement.py
├── combat.py
├── construction.py
└── movement.py
```

First, create new rule instance using predicates defined in `toolkit/caption/src/predicate.py`. The new rule has to be under `toolkit/caption/src/rule` directory.

```python
import src.predicates as P
from src.engine import Rule

NewRule = Rule(
    name="NewRule",
    checker=P.Kill("mob"),
    templates=["kill a mob"],
)
```

Second, register the new rule to the Rule Set in `toolkit/caption/src/rule/__init__.py`:

```python
RULES = [
    achievement.Harvest,
    ...
    NewRule, # New Rule Here
]
