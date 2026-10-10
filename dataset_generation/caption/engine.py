import numpy as np
from itertools import product

from dataset_generation.caption.common import EnvState
from dataset_generation.caption.predicate import Predicate

VALID_STEP = {
    "Achievement": {
        "Harvest": 1,
        "Construct": 1,
        "Craft": 1,
        "Kill": 1,
        "Sleep": 1,
    },
    "Movement": {
        "Stay": 5,
        "Move Towards": 5,
        "Approach": 5,
        "Flee": 5,
        "Explore": 5,
    },
    "Construction": {
        "Build Shelter": 1,
        "Build Path Over": 2,
        "Dig Tunnel": 1,
    },
    "Combat": {
        "Block Attack": 1,
        "Attacked": 1,
    },
}

# Minimum number of steps required in the sequence for an action to be considered valid.
MIN_STEP_REQUIRED = {
    category: {action: min(steps, (steps // 2) + 1) for action, steps in actions.items()}
    for category, actions in VALID_STEP.items()
}


def combine_bindings(bindings: list[dict[str, any]]) -> dict[str, any]:
    combined = {}
    for binding in bindings:
        for key, value in binding.items():
            if key in combined:
                current = combined[key]
                if current == value:
                    continue

                if not isinstance(current, list):
                    current = [current]

                if isinstance(value, list):
                    for v in value:
                        if v not in current:
                            current.append(v)
                else:
                    if value not in current:
                        current.append(value)
                combined[key] = current
            else:
                combined[key] = value

    for k, v in combined.items():
        if not isinstance(v, list):
            combined[k] = [v]

    return combined


class Rule:
    def __init__(
        self,
        name: str,
        checker: Predicate,
        templates: list[str],
        pivot_offset: int = 0,
    ):
        self.name = name
        self.checker = checker
        self.templates = templates
        self.pivot_offset = pivot_offset

    def apply(self, state: EnvState, action, next_state: EnvState) -> list[tuple[str, int]]:
        captions = []
        bindings = self.checker.instantiate(state, action, next_state)
        if bindings:
            combined = combine_bindings(bindings)

            if combined:
                keys, values = zip(*combined.items())
                for one_values in product(*values):
                    binding = {
                        k: v.name.replace("_", " ").lower() for k, v in zip(keys, one_values)
                    }
                    captions.append(self.templates[0].format(**binding))
            else:
                captions.append(self.templates[0])
        return [(caption, self.pivot_offset, self.name) for caption in captions]


class Engine:
    def __init__(self, rules: list[Rule]):
        self.rules = rules
        self.rule_stats = np.zeros(len(rules), dtype=np.uint64)

    @property
    def window_size(self) -> int:
        return max(len(rule.checker) for rule in self.rules)

    def generate(
        self,
        states: list[EnvState],
        actions: list,
        next_states: list[EnvState],
    ) -> list[tuple[str, int, str]]:
        captions = []
        for rule_id, rule in enumerate(self.rules):
            if len(rule.checker) > 1:
                window = len(rule.checker)
                if len(states) < window:
                    continue
                caption = rule.apply(states[-window:], actions[-window:], next_states[-window:])
            else:
                caption = rule.apply(states[-1], actions[-1], next_states[-1])

            if caption:
                self.rule_stats[rule_id] += 1
            captions.extend(caption)
        return captions

    @property
    def rule_statistics(self) -> dict[str, int]:
        return {rule.name: count for rule, count in zip(self.rules, self.rule_stats)}


def generate_caption_and_pivot(
    engine: Engine, env_states, actions, dense
) -> list[tuple[int, str, str]]:
    ep_len = len(actions)
    window_size = engine.window_size

    result = []
    for end_idx in range(ep_len):
        start_idx = max(0, end_idx - window_size)
        end_idx = end_idx + 1

        captions = engine.generate(
            env_states[start_idx:end_idx],
            actions[start_idx:end_idx],
            env_states[start_idx + 1 : end_idx + 1],
        )

        result.extend(
            [(end_idx + pivot_offset, caption, name) for (caption, pivot_offset, name) in captions]
        )

    if dense:
        result = generate_dense_captions(engine, result, ep_len)

    return result


def get_step(name: str, dict: dict) -> int:
    for category in dict.values():
        if name in category:
            return category[name]


def is_sole_name_valid(lst: list, window_size: int, rule_name: str) -> bool:
    unique_names = {t[2] for group in lst for t in group if t}

    if unique_names == {rule_name}:
        for index in [window_size - 1, window_size]:
            if not lst[index] or any(t[2] != rule_name for t in lst[index]):
                return False

    return True


def filter_captions(
    groups: list[tuple[str, int, str]],
    prev_names: list[str],
    filter_names: list[str],
) -> tuple[list[tuple[str, int, str]]]:
    """Filters out redundant captions from a group.

    Args:
        groups: A list of (timestamp, caption, name) tuples.
        prev_names: A list of previously encountered names.
        filter_names: Names to filter out.

    Returns:
        A tuple (filtered_group, updated_prev_names).
    """
    current_names = [name for _, _, name in groups]
    filtered = [
        (timestamp, caption, name)
        for timestamp, caption, name in groups
        if not (name in filter_names and name in prev_names)
    ]
    if filtered:
        prev_names = current_names
    return filtered, prev_names


def generate_windowed_captions(
    window: list[list[tuple[int, str, str]]],
    pivot: int,
    idxs: tuple[int, int, int],
) -> list[tuple[int, list[str], str]]:
    """Generates dense captions from a window of caption groups.

    Args:
        window: List of caption groups for each time step.
        pivot: The current pivot index.
        idxs: A tuple (window_size, start_idx, end_idx)

    Returns:
        A list of dense caption tuples: (pivot, [caption texts], last action name).
    """
    caption_groups, prev_names = [], []
    window_size, start_idx, end_idx = idxs

    movement_threshold = max(1, max(MIN_STEP_REQUIRED["Movement"].values()) - 1)
    movement_count = sum(
        1
        for group in window[start_idx:end_idx]
        if any(name in MIN_STEP_REQUIRED["Movement"] for _, _, name in group)
    )

    # Ignore movement captions if there are an insufficient number of movements.
    if 0 < movement_count < movement_threshold:
        return []

    # Filter out captions that don't meet the minimum transition steps.
    valid_captions = []
    for i, group in enumerate(window):
        valid_group = []
        for timestamp, caption, name in group:
            min_req = get_step(name, MIN_STEP_REQUIRED)
            if min_req is None:
                print(name, min_req)
            valid_step = get_step(name, VALID_STEP)

            if i < min_req:  # Check for unappropriate captions at the beginning
                continue
            elif (
                window_size + (valid_step - min_req) < i
            ):  # Check for unappropriate captions at the end
                continue

            valid_group.append((timestamp, caption, name))
        valid_captions.append(valid_group)

    if not is_sole_name_valid(valid_captions, window_size, "Stay"):
        return []

    # Apply further filtering to remove redundant captions.
    for group in valid_captions:
        filtered, prev_names = filter_captions(
            group, prev_names, filter_names=MIN_STEP_REQUIRED["Movement"].keys()
        )
        if not filtered:
            continue
        caption_groups.append(filtered)

    if not caption_groups:
        return []

    # Generate a Cartesian product of all_groups and compile the results
    # into dense caption tuples.
    dense_caption = []
    for combination in product(*caption_groups):
        caption_texts = [caption for _, caption, _ in combination]
        _, _, last_name = combination[-1]
        dense_caption.append((pivot, caption_texts, last_name))

    return dense_caption


def generate_dense_captions(engine: Engine, result, ep_len) -> list[tuple[int, list[str], str]]:
    window_size = engine.window_size
    extended_offset = max(
        0,
        window_size
        - max(step for actions in MIN_STEP_REQUIRED.values() for step in actions.values()),
    )

    captions_by_pivot = [[] for _ in range(ep_len + 1)]
    for pivot, caption, name in result:
        captions_by_pivot[pivot].append((pivot, caption, name))

    dense_caption: list[tuple[int, list[str], str]] = []

    for pivot in range(ep_len):
        start_idx = max(0, pivot - window_size)
        end_idx = pivot + 1
        extended_end_idx = min(ep_len, end_idx + extended_offset)

        sliding_window = captions_by_pivot[start_idx:extended_end_idx]
        idxs = (window_size, start_idx, end_idx)
        dense_caption.extend(generate_windowed_captions(sliding_window, pivot, idxs))

    return dense_caption
