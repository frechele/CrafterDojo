import numpy as np
from itertools import product
import random

from src.common import EnvState
from src.predicate import Predicate


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
    def __init__(self, name: str, checker: Predicate, templates: list[str], pivot_offset: int = 0):
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
                    binding = { k: v.name.replace('_', ' ').lower() for k, v in zip(keys, one_values) }
                    captions.append(random.choice(self.templates).format(**binding))
            else:
                captions.append(random.choice(self.templates))
        return [(caption, self.pivot_offset, self.name) for caption in captions]


class Engine:
    def __init__(self, rules: list[Rule]):
        self.rules = rules
        self.rule_stats = np.zeros(len(rules), dtype=np.uint64)

    @property
    def window_size(self) -> int:
        return max(len(rule.checker) for rule in self.rules)

    def generate(self, states: list[EnvState], actions: list, next_states: list[EnvState]) -> list[tuple[str, int, str]]:
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
        return { rule.name: count for rule, count in zip(self.rules, self.rule_stats) }
    

def generate_caption_and_pivot(engine: Engine, env_states, actions) -> list[tuple[int, str, str]]:
    ep_len = len(actions)
    window_size = engine.window_size

    result = []
    for end_idx in range(ep_len):
        start_idx = max(0, end_idx - window_size)
        end_idx = end_idx + 1

        captions = engine.generate(
            env_states[start_idx:end_idx],
            actions[start_idx:end_idx],
            env_states[start_idx+1:end_idx+1],
        )
        
        result.extend([(end_idx + pivot_offset, caption, name) for (caption, pivot_offset, name) in captions])
    
    return result


def get_step(name: str, dict: dict) -> int:
    for category in dict.values():
        if name in category:
            return category[name]
        
def is_sole_name_valid(lst: list, window_size: int, rule_name: str) -> bool:
    unique_names = {t[2] for group in lst for t in group if t}
   
    if unique_names == {rule_name}:
        for index in [window_size-1, window_size]:
            if not lst[index] or any(t[2] != rule_name for t in lst[index]):
                return False
       
    return True
