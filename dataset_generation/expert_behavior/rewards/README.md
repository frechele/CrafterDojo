# Reward Wrappers

## Built-in Rewards

| Name | Description | Parameters |
|------|-------------|------------|
| `crafter` | Pass-through for the base Craftax reward (achievement + health) | - |
| `survival` | Base reward + death/lava penalty | `death_penalty: int = 0` |
| `gatherer` | Base reward + bonus for surpassing inventory maximums | `alpha: float = 0.1` |
| `combat` | Base reward + bonus per zombie/skeleton killed | `alpha: float = 1.0` |

## CLI Usage

Use `--reward_wrappers` to specify wrappers. They are applied in list order (first = innermost, last = outermost).

```bash
# Base reward (default)
--reward_wrappers crafter

# Survival with death penalty
--reward_wrappers survival:death_penalty=10

# Gatherer with alpha
--reward_wrappers gatherer:alpha=0.2

# Gatherer + survival combined
--reward_wrappers gatherer:alpha=0.1 survival:death_penalty=5
```

Parameter format: `name:key=val,key2=val2`

## Adding a New Reward Function

### 1. Create a file

Add a new file under `dataset_generation/expert_behavior/rewards/`.

### 2. Subclass RewardWrapper and register

```python
from functools import partial
from typing import Union

import chex
import jax

from dataset_generation.expert_behavior.rewards.base import RewardWrapper
from dataset_generation.expert_behavior.rewards.registry import register_reward


@register_reward("my_reward")
class MyRewardWrapper(RewardWrapper):
    """One-line description."""

    def __init__(self, env, my_param: float = 1.0):
        super().__init__(env)
        self.my_param = my_param

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, key: chex.PRNGKey, params=None):
        obs, env_state = self._env.reset(key, params)
        # Optionally wrap in a custom state (see below)
        return obs, env_state

    @partial(jax.jit, static_argnums=(0, 4))
    def step(self, key: chex.PRNGKey, state, action: Union[int, float], params=None):
        obs, env_state, reward, done, info = self._env.step(key, state, action, params)
        # Modify the reward
        reward = reward + self.my_param * ...
        return obs, env_state, reward, done, info
```

### 3. Custom state (if needed)

If your wrapper needs to track state across steps, use `@struct.dataclass`.
You **must** implement `__getattr__` to proxy attribute access so that wrapper stacking works correctly.

```python
from typing import Any
from flax import struct

@struct.dataclass
class MyEnvState:
    env_state: Any
    my_tracking_field: ...

    def __getattr__(self, name):
        return getattr(self.env_state, name)
```

In `step`, pass `state.env_state` to the inner env and return a new `MyEnvState`.

### 4. Add import to `__init__.py`

```python
from dataset_generation.expert_behavior.rewards.my_reward import MyRewardWrapper as MyRewardWrapper
```

The `@register_reward` decorator runs at import time, so this automatically registers the wrapper.

### 5. Use it

```bash
--reward_wrappers my_reward:my_param=2.0
```

### Notes

- `@partial(jax.jit, static_argnums=...)` is required on `reset` and `step`
- Python `if` on JAX traced values is not allowed — use `jnp.where` instead
- `LogWrapper` is always applied automatically as the outermost wrapper for episode tracking
