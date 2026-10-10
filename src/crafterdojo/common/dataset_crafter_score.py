import glob
import os
from collections import OrderedDict

import numpy as np

from crafterdojo.common.crafter_score import compute_crafter_score_from_fraction


_FALLBACK_ACHIEVEMENT_NAMES = (
    "collect_wood",
    "place_table",
    "eat_cow",
    "collect_sapling",
    "collect_drink",
    "make_wood_pickaxe",
    "make_wood_sword",
    "place_plant",
    "defeat_zombie",
    "collect_stone",
    "place_stone",
    "eat_plant",
    "defeat_skeleton",
    "make_stone_pickaxe",
    "make_stone_sword",
    "wake_up",
    "place_furnace",
    "collect_coal",
    "collect_iron",
    "collect_diamond",
    "make_iron_pickaxe",
    "make_iron_sword",
)


def get_craftax_achievement_names() -> tuple[str, ...]:
    try:
        from craftax.craftax_classic.constants import Achievement
    except Exception:
        return _FALLBACK_ACHIEVEMENT_NAMES

    names = tuple(achievement.name.lower() for achievement in Achievement)
    return names or _FALLBACK_ACHIEVEMENT_NAMES


def _episode_ids_from_pattern(
    pattern: str,
    filename_prefix: str,
    filename_suffix: str,
) -> list[int]:
    episode_ids = []
    for path in sorted(glob.glob(pattern)):
        filename = os.path.basename(path)
        if not filename.startswith(filename_prefix) or not filename.endswith(
            filename_suffix
        ):
            continue
        episode_ids.append(
            int(filename.removeprefix(filename_prefix).removesuffix(filename_suffix))
        )
    return episode_ids


def list_dataset_episode_ids(dataset_root: str) -> list[int]:
    dataset_root = os.path.abspath(dataset_root)

    candidates = (
        ("state", "episode_", ".npz"),
        ("len", "len_", ".txt"),
        ("video", "video_", ".mp4"),
        ("action", "action_", ".pkl"),
    )
    for dirname, filename_prefix, filename_suffix in candidates:
        pattern = os.path.join(
            dataset_root,
            dirname,
            f"{filename_prefix}*{filename_suffix}",
        )
        episode_ids = _episode_ids_from_pattern(pattern, filename_prefix, filename_suffix)
        if episode_ids:
            return episode_ids

    raise FileNotFoundError(
        f"Could not infer episode ids under '{dataset_root}'. "
        "Expected one of state/, len/, video/, or action/."
    )


def resolve_state_dataset_root(dataset_root: str) -> str:
    dataset_root = os.path.abspath(dataset_root.rstrip("/"))
    if os.path.isdir(os.path.join(dataset_root, "state")):
        return dataset_root

    basename = os.path.basename(dataset_root)
    if "_filtered" in basename:
        raw_basename = basename.split("_filtered", 1)[0]
        candidate = os.path.join(os.path.dirname(dataset_root), raw_basename)
        if os.path.isdir(os.path.join(candidate, "state")):
            return candidate

    raise FileNotFoundError(
        f"Could not find a state/ directory for '{dataset_root}'. "
        "If this is a filtered dataset, keep the unfiltered raw dataset next to it."
    )


def _load_final_achievements(state_dataset_root: str, episode_id: int) -> np.ndarray:
    state_path = os.path.join(
        state_dataset_root,
        "state",
        f"episode_{episode_id:05d}.npz",
    )
    if not os.path.exists(state_path):
        raise FileNotFoundError(
            f"Missing state episode for episode_id={episode_id:05d}: '{state_path}'"
        )

    episode = np.load(state_path)
    try:
        achievements = np.asarray(episode["achievements"], dtype=bool)
    finally:
        episode.close()

    if achievements.ndim == 0:
        raise ValueError(
            f"Invalid achievements array in '{state_path}': "
            f"shape={achievements.shape}"
        )
    if achievements.ndim == 1:
        return achievements
    return achievements[-1]


def compute_dataset_crafter_score(dataset_root: str) -> dict:
    dataset_root = os.path.abspath(dataset_root)
    episode_ids = list_dataset_episode_ids(dataset_root)
    state_root = resolve_state_dataset_root(dataset_root)
    achievement_names = get_craftax_achievement_names()

    achievement_counts = np.zeros(len(achievement_names), dtype=np.int64)
    for episode_id in episode_ids:
        final_achievements = _load_final_achievements(state_root, episode_id)
        if final_achievements.shape[-1] != len(achievement_names):
            raise ValueError(
                f"Achievement shape mismatch for episode_{episode_id:05d}: "
                f"expected {len(achievement_names)}, got {final_achievements.shape[-1]}"
            )
        achievement_counts += final_achievements.astype(np.int64, copy=False)

    num_episodes = float(len(episode_ids))
    achievement_unlock_rate = OrderedDict(
        (name, float(count) / num_episodes)
        for name, count in zip(achievement_names, achievement_counts)
    )
    achievement_success_rate_percent = OrderedDict(
        (name, rate * 100.0) for name, rate in achievement_unlock_rate.items()
    )

    return {
        "dataset_root": dataset_root,
        "state_root": state_root,
        "episodes": int(num_episodes),
        "achievement_unlock_rate": achievement_unlock_rate,
        "achievement_success_rate_percent": achievement_success_rate_percent,
        "crafter_score": compute_crafter_score_from_fraction(achievement_unlock_rate),
    }
