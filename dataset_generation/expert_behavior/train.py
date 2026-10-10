import argparse
import os
import time
import warnings
from collections.abc import Mapping

import jax
import jax.numpy as jnp
import numpy as np
import optax

from flax.training import orbax_utils
from orbax.checkpoint import (
    PyTreeCheckpointer,
    CheckpointManagerOptions,
    CheckpointManager,
)

import wandb
from flax.training.train_state import TrainState

from craftax.craftax_env import make_craftax_env_from_name

from dataset_generation.expert_behavior.nn import (
    ActorCriticRNN,
    ScannedRNN,
    Transition,
)
from dataset_generation.expert_behavior.rewards import get_reward_wrapper
from dataset_generation.expert_behavior.rewards.log_wrapper import LogWrapper
from dataset_generation.expert_behavior.wrappers import (
    OptimisticResetVecEnvWrapper,
    BatchEnvWrapper,
    AutoResetEnvWrapper,
)
from dataset_generation.expert_behavior.logging import create_log_dict, batch_log


def _parse_wrapper_entry(entry: str) -> tuple[str, dict]:
    """Parse a wrapper entry string into (name, kwargs).

    Supports formats:
        "log"                          -> ("log", {})
        "survival:death_penalty=10"    -> ("survival", {"death_penalty": 10})
        "gatherer:alpha=0.2"           -> ("gatherer", {"alpha": 0.2})
        "gatherer:alpha=0.2,beta=0.5"  -> ("gatherer", {"alpha": 0.2, "beta": 0.5})
    """
    if ":" not in entry:
        return entry, {}
    name, params_str = entry.split(":", 1)
    kwargs = {}
    for param in params_str.split(","):
        key, val = param.split("=", 1)
        try:
            kwargs[key] = int(val)
        except ValueError:
            try:
                kwargs[key] = float(val)
            except ValueError:
                kwargs[key] = val
    return name, kwargs


_SPAWN_PARAM_KEYS = (
    "spawn_cow_chance",
    "spawn_zombie_base_chance",
    "spawn_zombie_night_chance",
    "spawn_skeleton_chance",
)


def _apply_env_param_overrides(env_params, config):
    overrides = {}
    for key in _SPAWN_PARAM_KEYS:
        val = config.get(key.upper())
        if val is not None:
            overrides[key] = float(val)
    if overrides:
        env_params = env_params.replace(**overrides)
    return env_params


def _apply_reward_wrappers(env, config):
    """Apply reward wrappers from config["REWARD_WRAPPERS"].

    Each entry is either:
    - A string: ``"name"`` or ``"name:key=val,key2=val2"``
    - A dict: ``{"name": ..., "kwargs": {...}}``
    """
    entries = config.get("REWARD_WRAPPERS", ["crafter"])
    for entry in entries:
        if isinstance(entry, str):
            name, kwargs = _parse_wrapper_entry(entry)
        else:
            name = entry["name"]
            kwargs = dict(entry.get("kwargs", {}))

        wrapper_cls = get_reward_wrapper(name)
        env = wrapper_cls(env, **kwargs)

    return env


def make_train(config):
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    config["MINIBATCH_SIZE"] = config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]

    # Create environment
    env = make_craftax_env_from_name(config["ENV_NAME"], not config["USE_OPTIMISTIC_RESETS"])
    env_params = _apply_env_param_overrides(env.default_params, config)

    # Apply reward wrappers (registry-based)
    env = _apply_reward_wrappers(env, config)

    # LogWrapper before batching — OptimisticResetVecEnvWrapper vmaps internally
    env = LogWrapper(env)

    # Wrap with a batcher, maybe using optimistic resets
    if config["USE_OPTIMISTIC_RESETS"]:
        env = OptimisticResetVecEnvWrapper(
            env,
            num_envs=config["NUM_ENVS"],
            reset_ratio=min(config["OPTIMISTIC_RESET_RATIO"], config["NUM_ENVS"]),
        )
    else:
        env = AutoResetEnvWrapper(env)
        env = BatchEnvWrapper(env, num_envs=config["NUM_ENVS"])

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        # INIT NETWORK
        network = ActorCriticRNN(env.action_space(env_params).n, config=config)
        rng, _rng = jax.random.split(rng)
        init_x = (
            jnp.zeros((1, config["NUM_ENVS"], *env.observation_space(env_params).shape)),
            jnp.zeros((1, config["NUM_ENVS"])),
        )
        init_hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], config["LAYER_SIZE"])
        network_params = network.init(_rng, init_hstate, init_x)
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        obsv, env_state = env.reset(_rng, env_params)
        init_hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], config["LAYER_SIZE"])

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                (
                    train_state,
                    env_state,
                    last_obs,
                    last_done,
                    hstate,
                    rng,
                    update_step,
                ) = runner_state
                rng, _rng = jax.random.split(rng)

                # SELECT ACTION
                ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
                hstate, pi, value = network.apply(train_state.params, hstate, ac_in)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                value, action, log_prob = (
                    value.squeeze(0),
                    action.squeeze(0),
                    log_prob.squeeze(0),
                )

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                obsv, env_state, reward, done, info = env.step(_rng, env_state, action, env_params)

                transition = Transition(last_done, action, value, reward, log_prob, last_obs, info)
                runner_state = (
                    train_state,
                    env_state,
                    obsv,
                    done,
                    hstate,
                    rng,
                    update_step,
                )
                return runner_state, transition

            initial_hstate = runner_state[-3]
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            (
                train_state,
                env_state,
                last_obs,
                last_done,
                hstate,
                rng,
                update_step,
            ) = runner_state
            ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
            _, _, last_val = network.apply(train_state.params, hstate, ac_in)
            last_val = last_val.squeeze(0)

            def _calculate_gae(traj_batch, last_val, last_done):
                def _get_advantages(carry, transition):
                    gae, next_value, next_done = carry
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - next_done) - value
                    gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - next_done) * gae
                    return (gae, value, done), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val, last_done),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val, last_done)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    init_hstate, traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, init_hstate, traj_batch, gae, targets):
                        # RERUN NETWORK
                        _, pi, value = network.apply(
                            params, init_hstate[0], (traj_batch.obs, traj_batch.done)
                        )
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(
                            -config["CLIP_EPS"], config["CLIP_EPS"]
                        )
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, init_hstate, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                (
                    train_state,
                    init_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                ) = update_state

                rng, _rng = jax.random.split(rng)
                permutation = jax.random.permutation(_rng, config["NUM_ENVS"])
                batch = (init_hstate, traj_batch, advantages, targets)

                shuffled_batch = jax.tree.map(lambda x: jnp.take(x, permutation, axis=1), batch)

                minibatches = jax.tree.map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(
                            x,
                            [x.shape[0], config["NUM_MINIBATCHES"], -1] + list(x.shape[2:]),
                        ),
                        1,
                        0,
                    ),
                    shuffled_batch,
                )

                train_state, total_loss = jax.lax.scan(_update_minbatch, train_state, minibatches)
                update_state = (
                    train_state,
                    init_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, total_loss

            init_hstate = initial_hstate[None, :]  # TBH
            update_state = (
                train_state,
                init_hstate,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = jax.tree.map(
                lambda x: (
                    (x * traj_batch.info["returned_episode"]).sum()
                    / traj_batch.info["returned_episode"].sum()
                ),
                traj_batch.info,
            )
            rng = update_state[-1]
            if config["DEBUG"] and config["USE_WANDB"]:

                def callback(metric, update_step):
                    to_log = create_log_dict(metric, config)
                    crafter_score = _compute_crafter_score(metric)
                    if crafter_score is not None:
                        to_log["crafter_score"] = crafter_score
                    batch_log(update_step, to_log, config)

                jax.debug.callback(callback, metric, update_step)

            runner_state = (
                train_state,
                env_state,
                last_obs,
                last_done,
                hstate,
                rng,
                update_step + 1,
            )
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            env_state,
            obsv,
            jnp.zeros((config["NUM_ENVS"]), dtype=bool),
            init_hstate,
            _rng,
            0,
        )
        runner_state, metric = jax.lax.scan(_update_step, runner_state, None, config["NUM_UPDATES"])
        return {"runner_state": runner_state, "metric": metric}

    return train


def _compute_crafter_score(info):
    achievement_values = []
    for key, value in info.items():
        if "achievements" not in key.lower():
            continue
        if isinstance(value, Mapping):
            for nested_value in value.values():
                nested_array = np.asarray(nested_value)
                if nested_array.size:
                    achievement_values.append(nested_array.reshape(-1))
        else:
            array = np.asarray(value)
            if array.size:
                achievement_values.append(array.reshape(-1))

    if not achievement_values:
        return None

    percents = np.concatenate(achievement_values).astype(np.float32, copy=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return float(np.exp(np.nanmean(np.log1p(percents))) - 1)


def large_number_to_str(num):
    if num >= 1e12:
        return f"{int(num / 1e12)}T"
    if num >= 1e9:
        return f"{int(num / 1e9)}B"
    if num >= 1e6:
        return f"{int(num / 1e6)}M"
    if num >= 1e3:
        return f"{int(num / 1e3)}K"

    return f"{int(num)}"


def run_ppo(config):
    config = {k.upper(): v for k, v in config.__dict__.items()}

    config.setdefault("REWARD_WRAPPERS", ["crafter"])
    config.setdefault("WANDB_PROJECT", "crafterdojo-playdata")

    if config["USE_WANDB"]:
        reward_tags = [
            _parse_wrapper_entry(e)[0] if isinstance(e, str) else e["name"]
            for e in config["REWARD_WRAPPERS"]
        ]
        run_name = "-".join(reward_tags) + "-PPO_RNN-" + large_number_to_str(config["TOTAL_TIMESTEPS"])
        wandb_dir = os.path.abspath(os.path.join("wandb", run_name))
        os.makedirs(wandb_dir, exist_ok=True)
        wandb.init(
            project=config["WANDB_PROJECT"],
            entity=config["WANDB_ENTITY"],
            config=config,
            name=run_name,
            tags=reward_tags,
            dir=wandb_dir,
        )

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_REPEATS"])

    train_jit = jax.jit(make_train(config))
    train_vmap = jax.vmap(train_jit)

    t0 = time.time()
    out = train_vmap(rngs)
    t1 = time.time()
    print("Time to run experiment", t1 - t0)
    print("SPS: ", config["TOTAL_TIMESTEPS"] / (t1 - t0))

    if config["USE_WANDB"]:

        def _save_network(rs_index, dir_name):
            train_states = out["runner_state"][rs_index]
            train_state = jax.tree.map(lambda x: x[0], train_states)
            orbax_checkpointer = PyTreeCheckpointer()
            options = CheckpointManagerOptions(max_to_keep=1, create=True)
            path = os.path.join(wandb.run.dir, dir_name)
            checkpoint_manager = CheckpointManager(path, orbax_checkpointer, options)
            print(f"saved runner state to {path}")
            save_args = orbax_utils.save_args_from_target(train_state)
            checkpoint_manager.save(
                int(config["TOTAL_TIMESTEPS"]),
                train_state,
                save_kwargs={"save_args": save_args},
            )

        if config["SAVE_POLICY"]:
            _save_network(0, "policies")


def main_cli(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="Craftax-Classic-Symbolic-v1")
    parser.add_argument("--num_envs", type=int, default=1024)
    parser.add_argument("--total_timesteps", type=lambda x: int(float(x)), default=1e9)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num_steps", type=int, default=64)
    parser.add_argument("--update_epochs", type=int, default=4)
    parser.add_argument("--num_minibatches", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.8)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--activation", type=str, default="tanh")
    parser.add_argument("--anneal_lr", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--jit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=np.random.randint(2**31))
    parser.add_argument("--use_wandb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_policy", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--num_repeats", type=int, default=1)
    parser.add_argument("--layer_size", type=int, default=512)
    parser.add_argument("--wandb_project", type=str, default="crafterdojo-playdata")
    parser.add_argument("--wandb_entity", type=str)
    parser.add_argument(
        "--use_optimistic_resets", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--optimistic_reset_ratio", type=int, default=16)
    parser.add_argument("--death_penalty", type=int, default=10)
    parser.add_argument(
        "--reward_wrappers",
        nargs="*",
        default=None,
        help="Reward wrapper names to apply in order (e.g., 'gatherer log').",
    )
    parser.add_argument("--spawn_cow_chance", type=float, default=None)
    parser.add_argument("--spawn_zombie_base_chance", type=float, default=None)
    parser.add_argument("--spawn_zombie_night_chance", type=float, default=None)
    parser.add_argument("--spawn_skeleton_chance", type=float, default=None)

    args, rest_args = parser.parse_known_args(argv)
    if rest_args:
        raise ValueError(f"Unknown args {rest_args}")

    if args.seed is None:
        args.seed = np.random.randint(2**31)

    if args.jit:
        run_ppo(args)
    else:
        with jax.disable_jit():
            run_ppo(args)


if __name__ == "__main__":
    main_cli()
