import jax
import jax.numpy as jnp
import numpy as np
from orbax.checkpoint import PyTreeCheckpointer, CheckpointManager, CheckpointManagerOptions
import optax
import argparse
from craftax.craftax_env import make_craftax_env_from_name
from flax.training.train_state import TrainState
import os
import yaml
from tqdm import tqdm

from src.train import ActorCriticRNN, ScannedRNN


def load_policy(checkpoint_path, config, env, action_dim):
    orbax_checkpointer = PyTreeCheckpointer()
    options = CheckpointManagerOptions(max_to_keep=1, create=False)
    policies_path = os.path.join(checkpoint_path, "policies")
    policies_path = os.path.abspath(policies_path)
    checkpoint_manager = CheckpointManager(
        policies_path, orbax_checkpointer, options
    )

    """Load a trained policy from a checkpoint"""
    network = ActorCriticRNN(action_dim=action_dim, config=config)

    env_params = env.default_params

    init_x = (
        jnp.zeros(
            (1, config["NUM_ENVS"], *env.observation_space(env_params).shape),
        ),
        jnp.zeros((1, config["NUM_ENVS"])),
    )
    init_hstate = ScannedRNN.initialize_carry(
        config["NUM_ENVS"], config["LAYER_SIZE"]
    )

    rng = jax.random.PRNGKey(np.random.randint(2**31))
    rng, _rng, __rng = jax.random.split(rng, 3)
    network_params = network.init(_rng, init_hstate, init_x)

    step = checkpoint_manager.latest_step()
    if step is None:
        raise FileNotFoundError(f'No checkpoints found in {checkpoint_path}')

    # Create a dummy TrainState to match checkpoint structure
    if config.get("ANNEAL_LR", False):
        num_updates = config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
        linear_schedule = optax.linear_schedule(
            init_value=config["LR"],
            end_value=0.0,
            transition_steps=num_updates
        )
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

    train_state = checkpoint_manager.restore(step, items=train_state)
    
    # Extract only params for inference
    return network, train_state.params


def generate_trajectory(config, jit_apply, jit_step, params, env, env_params, rng, max_steps=1000):
    """Generate a trajectory using a trained policy"""
    obs, env_state = env.reset(key=rng)
    done = jnp.zeros(1, dtype=bool)
    init_hstate = ScannedRNN.initialize_carry(1, config["LAYER_SIZE"])
    hstate = init_hstate

    env_states = []
    actions_taken = []

    for _ in range(max_steps):
        obs_expanded = jnp.expand_dims(obs, axis=0)
        ac_in = (obs_expanded[np.newaxis, :], done[np.newaxis, :])

        # Get action from policy
        hstate, pi, value = jit_apply(params, hstate, ac_in)
        rng, _rng = jax.random.split(rng)
        action = pi.sample(seed=_rng).reshape(1)

        if action is not None:
            rng, _rng = jax.random.split(rng)
            env_states.append(env_state)
            actions_taken.append(action[0])
            
            obs, env_state, reward, done, info = jit_step(
                _rng, env_state, action[0], env_params
            )
            done = done[np.newaxis]

            if done[0]:
                break

    env_states.append(env_state)

    return jnp.array(actions_taken), env_states


def save_episode(state_path: str, actions, env_states):
    action_arr = np.array(actions)

    map_arr = np.stack([env_state.map for env_state in env_states], axis=0)
    mob_map_arr = np.stack([env_state.mob_map for env_state in env_states], axis=0)

    player_position_arr = np.stack([env_state.player_position for env_state in env_states], axis=0)
    player_direction_arr = np.array([env_state.player_direction for env_state in env_states])

    player_health_arr = np.array([env_state.player_health for env_state in env_states])
    player_food_arr = np.array([env_state.player_food for env_state in env_states])
    player_drink_arr = np.array([env_state.player_drink for env_state in env_states])
    player_energy_arr = np.array([env_state.player_energy for env_state in env_states])
    is_sleeping_arr = np.array([env_state.is_sleeping for env_state in env_states])

    player_recover_arr = np.array([env_state.player_recover for env_state in env_states])
    player_hunger_arr = np.array([env_state.player_hunger for env_state in env_states])
    player_thirst_arr = np.array([env_state.player_thirst for env_state in env_states])
    player_fatigue_arr = np.array([env_state.player_fatigue for env_state in env_states])

    inventory_wood_arr = np.array([env_state.inventory.wood for env_state in env_states])
    inventory_stone_arr = np.array([env_state.inventory.stone for env_state in env_states])
    inventory_coal_arr = np.array([env_state.inventory.coal for env_state in env_states])
    inventory_iron_arr = np.array([env_state.inventory.iron for env_state in env_states])
    inventory_diamond_arr = np.array([env_state.inventory.diamond for env_state in env_states])
    inventory_sapling_arr = np.array([env_state.inventory.sapling for env_state in env_states])
    inventory_wood_pickaxe_arr = np.array([env_state.inventory.wood_pickaxe for env_state in env_states])
    inventory_stone_pickaxe_arr = np.array([env_state.inventory.stone_pickaxe for env_state in env_states])
    inventory_iron_pickaxe_arr = np.array([env_state.inventory.iron_pickaxe for env_state in env_states])
    inventory_wood_sword_arr = np.array([env_state.inventory.wood_sword for env_state in env_states])
    inventory_stone_sword_arr = np.array([env_state.inventory.stone_sword for env_state in env_states])
    inventory_iron_sword_arr = np.array([env_state.inventory.iron_sword for env_state in env_states])

    zombies_position_arr = np.stack([env_state.zombies.position for env_state in env_states], axis=0)
    zombies_health_arr = np.array([env_state.zombies.health for env_state in env_states])
    zombies_mask_arr = np.array([env_state.zombies.mask for env_state in env_states])
    zombies_attack_cooldown_arr = np.array([env_state.zombies.attack_cooldown for env_state in env_states])

    cows_position_arr = np.stack([env_state.cows.position for env_state in env_states], axis=0)
    cows_health_arr = np.array([env_state.cows.health for env_state in env_states])
    cows_mask_arr = np.array([env_state.cows.mask for env_state in env_states])
    cows_attack_cooldown_arr = np.array([env_state.cows.attack_cooldown for env_state in env_states])

    skeletons_position_arr = np.stack([env_state.skeletons.position for env_state in env_states], axis=0)
    skeletons_health_arr = np.array([env_state.skeletons.health for env_state in env_states])
    skeletons_mask_arr = np.array([env_state.skeletons.mask for env_state in env_states])
    skeletons_attack_cooldown_arr = np.array([env_state.skeletons.attack_cooldown for env_state in env_states])

    arrows_position_arr = np.stack([env_state.arrows.position for env_state in env_states], axis=0)
    arrows_health_arr = np.array([env_state.arrows.health for env_state in env_states])
    arrows_mask_arr = np.array([env_state.arrows.mask for env_state in env_states])
    arrows_attack_cooldown_arr = np.array([env_state.arrows.attack_cooldown for env_state in env_states])

    arrow_directions_arr = np.stack([env_state.arrow_directions for env_state in env_states], axis=0)

    growing_plants_positions_arr = np.stack([env_state.growing_plants_positions for env_state in env_states], axis=0)
    growing_plants_age_arr = np.stack([env_state.growing_plants_age for env_state in env_states], axis=0)
    growing_plants_mask_arr = np.stack([env_state.growing_plants_mask for env_state in env_states], axis=0)

    light_level_arr = np.array([env_state.light_level for env_state in env_states])

    achievements_arr = np.stack([env_state.achievements for env_state in env_states], axis=0)

    state_rng_arr = np.stack([env_state.state_rng for env_state in env_states], axis=0)

    timestep_arr = np.array([env_state.timestep for env_state in env_states])

    np.savez_compressed(
        state_path,
        **{k[:-4]: v for k, v in locals().items() if k.endswith("_arr")},
    )


def main(args):
    root_path = args.save_path
    os.makedirs(root_path, exist_ok=True)
    state_path = os.path.join(root_path, "state")
    os.makedirs(state_path, exist_ok=True)

    with open(os.path.join(args.checkpoint_path, "config.yaml")) as f:
        raw_config = yaml.load(f, Loader=yaml.Loader)

        config = {}
        for key, value in raw_config.items():
            if isinstance(value, dict) and "value" in value:
                config[key] = value["value"]
    config["NUM_ENVS"] = 1
    del config["_wandb"]

    # Initialize environment
    env = make_craftax_env_from_name("Craftax-Classic-Symbolic-v1", False)
    env_params = env.default_params
    action_dim = env.action_space(env_params).n

    # Load policy
    network, params = load_policy(args.checkpoint_path, config, env, action_dim)

    jit_apply = jax.jit(network.apply)
    jit_step = jax.jit(env.step, static_argnums=(3,))

    # Generate trajectories
    for i in tqdm(range(args.start_ep, args.start_ep + args.n_episodes), total=args.n_episodes, desc="Generating trajectories"):
        _rng = jax.random.PRNGKey(args.seed + i * 10000)
        actions, env_states = generate_trajectory(
            config,
            jit_apply,
            jit_step,
            params, 
            env, 
            env_params, 
            _rng,
            args.max_steps
        )
        save_episode(
            os.path.join(state_path, f"episode_{i:05d}.npz"),
            actions,
            env_states
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True, 
                      help="Path to the trained policy checkpoint")
    parser.add_argument("--max_steps", type=int, default=10000,
                      help="Maximum steps per trajectory")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layer_size", type=int, default=512)
    parser.add_argument("--disable_jit", action="store_true")

    parser.add_argument("--save_path", type=str, default="../../CrafterDojo_Data")
    parser.add_argument("--start_ep", type=int, default=0)
    parser.add_argument("--n_episodes", type=int, default=20000,
                      help="Number of trajectories to generate")

    args = parser.parse_args()

    if args.disable_jit:
        print("[INFO] Disabling JIT")
        with jax.disable_jit():
            main(args)
    else:
        main(args)
