from dataset_generation.expert_behavior.wrappers.base import GymnaxWrapper


class RewardWrapper(GymnaxWrapper):
    """Base class for reward-shaping wrappers.

    Subclasses must implement reset() and step() following the Gymnax signature.
    They receive the base reward from the inner env and return a modified reward.
    """

    pass
