# CrafterDojo Expert Behavior Generator Toolkit

The Expert Behavior Generator toolkit trains an expert policy using reinforcement learning, then deploys it to generate large-scale synthetic demonstrations at scale. This toolkit is based on the PPO-RNN agent from [here](https://github.com/MichaelTMatthews/Craftax_Baselines).

## Quick Start

To generate the CrafterPlay dataset, execute the following commands:

```bash
# Step 1: Train PPO-RNN agent
uv run bash scripts/train_ppo.sh

# Step 2: Generate episodes (generate state pickle files)
uv run generate_trajectory.py --checkpoint_path <path>

# Step 3: Process episodes (generate video and action files)
uv run bash scripts/process_episodes.sh

# Step 4: Generate null actions
uv run bash scripts/nullaction_filtering.sh
```

## Customize Reward Function

To customize expert policy behavior, the reward function should be modified. First, please refer `DeathPenaltyWrapper` class in `toolkit/expert/src/wrappers.py`, and create a new wrapper class for your own reward function:

```python
class NewRewardWrapper(GymnaxWrapper):
    def __init__(self, env):
        super().__init__(env)

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, key: chex.PRNGKey, params=None):
        return self._env.reset(key, params)

    @partial(jax.jit, static_argnums=(0, 4))
    def step(
        self,
        key: chex.PRNGKey,
        state,
        action: Union[int, float],
        params=None,
    ):
        obs, env_state, reward, done, info = self._env.step(
            key, state, action, params
        )

        reward = reward - 1

        return obs, env_state, reward, done, info
```


Second, wrap the environment instance at `toolkit/expert/src/train.py` (line 148) like:

```python
env = NewRewardWrapper(env) # new reward function
env = DeathPenaltyWrapper(env, death_penalty=config["DEATH_PENALTY"]) # this can be removed
env = LogWrapper(env)
```

Lastly, generate a new expert behavior demonstration dataset by following [the generation process](#quick-start).
