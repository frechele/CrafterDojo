# Generating Custom Data with the CrafterDojo Toolkits

CrafterDojo contains the exact toolkits that produced the CrafterPlay and
CrafterCaption datasets:

1. **Expert Behavior Generator** (`dataset_generation/expert_behavior`; Section 4.1 of the
   paper): `train` → `generate` → `process`, plus a `filter` stage (Section 4.2 of the paper)
   used to prepare the C-VPT training data.
2. **Caption Generator** (`dataset_generation/caption`; Section 5.1 of the paper): a rule-based
   caption generation system (15 rules across four categories) over the generated environment
   states.

![Dataset generation pipeline — the Expert Behavior Generator produces CrafterPlay and the Caption Generator produces CrafterCaption](figures/generation_pipeline.png)

All commands below run from a scratch directory (stages write their outputs, including
aggregate statistics files, into the current working directory):

```bash
mkdir -p tutorial_output && cd tutorial_output
```

## 1 — Train an expert

`train` trains a PPO expert in `Craftax-Classic-Symbolic-v1`. The reward wrappers select the
expert: the ones below give the gatherer expert, and the survival expert uses only
`survival:death_penalty=10`. `--save_policy` stores the trained policy inside the wandb run
directory.

```bash
python -m dataset_generation.main expert train \
    --total_timesteps 1e5 --num_envs 32 --num_minibatches 4 \
    --reward_wrappers gatherer:alpha=0.1,place_penalty=1 survival:death_penalty=10 \
    --seed 0 --save_policy --no-debug
```

When the run finishes, its `files/` directory holds the policy (`policies/`) and the training
config (`config.yaml`, written by wandb) — exactly the `--checkpoint_path` layout `generate`
expects:

```bash
CKPT=$(ls -d wandb/gatherer-survival-PPO_RNN-100K/wandb/run-*/files | tail -1)
```

## 2 — Generate episodes

`generate` rolls out the trained policy in batched JAX environments and writes one compressed
NumPy archive per episode (`state/episode_<id>.npz`) with the full environment state
at every step.

```bash
python -m dataset_generation.main expert generate \
    --checkpoint_path "$CKPT" \
    --save_path custom_data \
    --n_episodes 3 --batch_size 3 --max_steps 10000 --seed 7
```

The remaining stages all operate on the generated split root:

```bash
ROOT=custom_data/Craftax-Classic-Symbolic-v1
```

## 3 — Render videos and extract actions

`process` renders each state sequence to a 144×144 MP4 and writes the action and
episode-length files (`video/`, `action/`, `len/`).

```bash
python -m dataset_generation.main expert process \
    --root "$ROOT" \
    --n_episodes 3 --save_workers 2 --load_workers 2
```

## 4 — Generate captions (CrafterCaption)

The Caption Generator monitors the environment states and emits caption segments
(`caption/episode_<id>.pkl`) using the 15 rules described in Section 5.1 of the paper.

```bash
python -m dataset_generation.main caption --root "$ROOT" --n_episodes 3
```

## 5 — Noop filtering (Section 4.2 of the paper)

`filter` applies the noop filtering described in Section 4.2 of the paper: noop sequences
shorter than 20 consecutive steps and actions issued while sleeping are removed, while longer
noop sequences are preserved (they can represent strategic waiting). The released datasets
ship unfiltered, so this is also the command that prepares the C-VPT training data from a
downloaded split — it only needs the `action/`, `video/`, and `len/` fields (pass
`--n_episodes 20000` for a full split).

```bash
python -m dataset_generation.main expert filter \
    --root "$ROOT" \
    --filter 20 --start_ep 0 --n_episodes 3
```

The filtered output lands in the sibling directory `<root>_filtered_20/` with filtered
`video/`, `action/`, and `len/` fields (release layout, so the same loaders apply), plus
`map/map_<id>.pkl` — a per-episode list mapping each raw frame index to its filtered index
(`-1` = frame dropped), so captions and states — which are indexed on the raw frames — stay
alignable with the filtered video.


## 6 — Custom rewards (CrafterPlay)

The reward wrappers control what the expert learns, and therefore what behavior the generated
data contains. To add your own, drop a file into
`dataset_generation/expert_behavior/rewards/`, register the class, and import it in the
package `__init__.py` (`dataset_generation/expert_behavior/rewards/README.md` covers details
such as tracking state across steps):

```python
# dataset_generation/expert_behavior/rewards/my_reward.py
from functools import partial

import jax

from dataset_generation.expert_behavior.rewards.base import RewardWrapper
from dataset_generation.expert_behavior.rewards.registry import register_reward


@register_reward("my_reward")
class MyRewardWrapper(RewardWrapper):
    def __init__(self, env, alpha: float = 1.0):
        super().__init__(env)
        self.alpha = alpha

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, key, params=None):
        return self._env.reset(key, params)

    @partial(jax.jit, static_argnums=(0, 4))
    def step(self, key, state, action, params=None):
        obs, env_state, reward, done, info = self._env.step(key, state, action, params)
        reward = reward + self.alpha * 0.0  # shape the reward here
        return obs, env_state, reward, done, info
```

Train with it — the rest of the pipeline is unchanged (the wandb run directory is named after
the wrapper list, so adjust the `CKPT` glob accordingly):

```bash
python -m dataset_generation.main expert train \
    --total_timesteps 1e5 --num_envs 32 --num_minibatches 4 \
    --reward_wrappers my_reward:alpha=2.0 survival:death_penalty=10 \
    --seed 0 --save_policy --no-debug
```

## 7 — Custom caption rules (CrafterCaption)

A caption rule pairs a *predicate* — a check over a state transition, or a window of
consecutive transitions — with a caption template whose `{placeholders}` are filled by the
variables the predicate binds. Predicates live in `dataset_generation/caption/predicate.py`
and compose with `And` / `Or` / `Not` / `Sequence`. For example, a rule that captions kills
happening at night:

```python
# dataset_generation/caption/rules/my_rules.py
from dataset_generation.caption import predicate as P
from dataset_generation.caption.engine import Rule

NightKill = Rule(
    name="Night Kill",
    checker=P.And(P.Kill("mob"), P.Not(P.IsDay())),
    templates=["kill {mob} at night"],
)
```

`P.Kill("mob")` matches a step where the agent attacks a mob and binds `{mob}` to it
("kill zombie at night", "kill cow at night", ...); passing a fixed value instead
(`P.Kill(P.Mob.ZOMBIE)`) restricts the rule to that mob. For checks the predicate library
does not cover, subclass `Predicate` and implement `instantiate`.

Register the rule by adding it to `RULES` in `dataset_generation/caption/rules/__init__.py`:

```python
from dataset_generation.caption.rules import my_rules

RULES = [
    # ...
    my_rules.NightKill,
]
```

Re-running the caption stage (Section 4) then writes the new rule's segments into
`caption/episode_<id>.pkl` alongside the existing ones, tagged with the rule name.
