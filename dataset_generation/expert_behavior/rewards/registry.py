from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dataset_generation.expert_behavior.rewards.base import RewardWrapper

_REWARD_REGISTRY: dict[str, type[RewardWrapper]] = {}


def register_reward(name: str):
    """Decorator to register a reward wrapper class under a given name.

    Usage::

        @register_reward("crafter")
        class CrafterRewardWrapper(RewardWrapper):
            ...
    """

    def decorator(cls):
        if name in _REWARD_REGISTRY:
            raise ValueError(
                f"Reward wrapper {name!r} is already registered "
                f"({_REWARD_REGISTRY[name].__qualname__})"
            )
        _REWARD_REGISTRY[name] = cls
        return cls

    return decorator


def get_reward_wrapper(name: str) -> type[RewardWrapper]:
    """Look up a reward wrapper class by its registered name."""
    if name not in _REWARD_REGISTRY:
        available = ", ".join(sorted(_REWARD_REGISTRY)) or "(none)"
        raise KeyError(f"Unknown reward wrapper: {name!r}. Available: {available}")
    return _REWARD_REGISTRY[name]


def list_reward_wrappers() -> list[str]:
    """Return the names of all registered reward wrappers."""
    return sorted(_REWARD_REGISTRY)
