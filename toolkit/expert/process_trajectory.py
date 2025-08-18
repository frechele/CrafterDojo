import argparse
import os
import glob
import pickle
import numpy as np
import cv2
import jax
import jax.numpy as jnp
from craftax.craftax_classic.renderer import render_craftax_pixels
from craftax.craftax_classic.envs.craftax_state import EnvState, Inventory, Mobs
from tqdm import tqdm
from craftax.craftax_classic.constants import Action

from src import rules as R


def is_sleeping(state: EnvState, action):
    return state.is_sleeping


def is_null_do_crafting(state: EnvState, action):
    is_crafting_wood_pickaxe = R.rule_craft_wood_pickaxe(state, action)
    is_crafting_stone_pickaxe = R.rule_craft_stone_pickaxe(state, action)
    is_crafting_iron_pickaxe = R.rule_craft_iron_pickaxe(state, action)

    is_crafting_wood_sword = R.rule_craft_wood_sword(state, action)
    is_crafting_stone_sword = R.rule_craft_stone_sword(state, action)
    is_crafting_iron_sword = R.rule_craft_iron_sword(state, action)

    return jnp.logical_not(
        jnp.logical_or(
            is_crafting_wood_pickaxe,
            jnp.logical_or(
                is_crafting_stone_pickaxe,
                jnp.logical_or(
                    is_crafting_iron_pickaxe,
                    jnp.logical_or(
                        is_crafting_wood_sword,
                        jnp.logical_or(
                            is_crafting_stone_sword,
                            is_crafting_iron_sword,
                        )
                    )
                )
            )
        )
    )


def is_null_do_action(state: EnvState, action):
    # Mobs
    is_attacking_zombie = R.rule_attack_zombie(state, action)
    is_attacking_cow = R.rule_attack_cow(state, action)
    is_attacking_skeleton = R.rule_attack_skeleton(state, action)

    is_attacking_mob = jnp.logical_or(
        is_attacking_zombie,
        jnp.logical_or(
            is_attacking_cow,
            is_attacking_skeleton,
        )
    )

    # Blocks
    is_mining_tree = R.rule_mine_tree(state, action)
    is_mining_stone = R.rule_mine_stone(state, action)
    is_mining_coal = R.rule_mine_coal(state, action)
    is_mining_iron = R.rule_mine_iron(state, action)
    is_mining_diamond = R.rule_mine_diamond(state, action)
    is_mining_sapling = R.rule_mine_sapling(state, action)

    is_mining_block = jnp.logical_or(
        is_mining_tree,
        jnp.logical_or(
            is_mining_stone,
            jnp.logical_or(
                is_mining_coal,
                jnp.logical_or(
                    is_mining_iron,
                    jnp.logical_or(
                        is_mining_diamond,
                        is_mining_sapling,
                    )
                )
            )
        )
    )

    is_drinking_water = R.rule_mine_water(state, action)
    is_eating_plant = R.rule_mine_plant(state, action)

    is_eating_something = jnp.logical_or(
        is_drinking_water,
        is_eating_plant,
    )

    return jnp.logical_not(
        jnp.logical_or(
            is_attacking_mob,
            jnp.logical_or(
                is_mining_block,
                is_eating_something,
            )
        )
    )


def is_null_place_block(state: EnvState, action):
    is_placing_crafting_table = R.rule_place_crafting_table(state, action)
    is_placing_furnace = R.rule_place_furnace(state, action)
    is_placing_stone = R.rule_place_stone(state, action)
    is_placing_sapling = R.rule_place_sapling(state, action)

    return jnp.logical_not(
        jnp.logical_or(
            is_placing_crafting_table,
            jnp.logical_or(
                is_placing_furnace,
                jnp.logical_or(
                    is_placing_stone,
                    is_placing_sapling,
                )
            )
        )
    )


def is_move_player(state: EnvState, action):
    return jnp.logical_or(
        action == Action.UP.value,
        jnp.logical_or(
            action == Action.DOWN.value,
            jnp.logical_or(
                action == Action.LEFT.value,
                action == Action.RIGHT.value,
            )
        )
    )


def is_sleeping_action(state: EnvState, action):
    return R.is_sleep(state, action)


def load_episode(npz_file: str):
    episode = np.load(npz_file)
    episode = { k: np.array(v) for k, v in episode.items() }

    actions = episode["action"]
    env_states = EnvState(
        map=jnp.array(episode["map"]),
        mob_map=jnp.array(episode["mob_map"]),

        player_position=jnp.array(episode["player_position"]),
        player_direction=jnp.array(episode["player_direction"]),
        player_health=jnp.array(episode["player_health"]),
        player_food=jnp.array(episode["player_food"]),
        player_drink=jnp.array(episode["player_drink"]),
        player_energy=jnp.array(episode["player_energy"]),
        is_sleeping=jnp.array(episode["is_sleeping"]),

        player_recover=jnp.array(episode["player_recover"]),
        player_hunger=jnp.array(episode["player_hunger"]),
        player_thirst=jnp.array(episode["player_thirst"]),
        player_fatigue=jnp.array(episode["player_fatigue"]),

        inventory=Inventory(
            wood=jnp.array(episode["inventory_wood"]),
            stone=jnp.array(episode["inventory_stone"]),
            coal=jnp.array(episode["inventory_coal"]),
            iron=jnp.array(episode["inventory_iron"]),
            diamond=jnp.array(episode["inventory_diamond"]),
            sapling=jnp.array(episode["inventory_sapling"]),
            wood_pickaxe=jnp.array(episode["inventory_wood_pickaxe"]),
            stone_pickaxe=jnp.array(episode["inventory_stone_pickaxe"]),
            iron_pickaxe=jnp.array(episode["inventory_iron_pickaxe"]),
            wood_sword=jnp.array(episode["inventory_wood_sword"]),
            stone_sword=jnp.array(episode["inventory_stone_sword"]),
            iron_sword=jnp.array(episode["inventory_iron_sword"]),
        ),

        zombies=Mobs(
            position=jnp.array(episode["zombies_position"]),
            health=jnp.array(episode["zombies_health"]),
            mask=jnp.array(episode["zombies_mask"]),
            attack_cooldown=jnp.array(episode["zombies_attack_cooldown"]),
        ),
        cows=Mobs(
            position=jnp.array(episode["cows_position"]),
            health=jnp.array(episode["cows_health"]),
            mask=jnp.array(episode["cows_mask"]),
            attack_cooldown=jnp.array(episode["cows_attack_cooldown"]),
        ),
        skeletons=Mobs(
            position=jnp.array(episode["skeletons_position"]),
            health=jnp.array(episode["skeletons_health"]),
            mask=jnp.array(episode["skeletons_mask"]),
            attack_cooldown=jnp.array(episode["skeletons_attack_cooldown"]),
        ),
        arrows=Mobs(
            position=jnp.array(episode["arrows_position"]),
            health=jnp.array(episode["arrows_health"]),
            mask=jnp.array(episode["arrows_mask"]),
            attack_cooldown=jnp.array(episode["arrows_attack_cooldown"]),
        ),
        arrow_directions=jnp.array(episode["arrow_directions"]),

        growing_plants_positions=jnp.array(episode["growing_plants_positions"]),
        growing_plants_age=jnp.array(episode["growing_plants_age"]),
        growing_plants_mask=jnp.array(episode["growing_plants_mask"]),

        light_level=jnp.array(episode["light_level"]),

        achievements=jnp.array(episode["achievements"]),

        state_rng=jnp.array(episode["state_rng"]),

        timestep=jnp.array(episode["timestep"]),
    )
    return actions, env_states


def main(args):
    def render_and_process(state, size):
        rendered = render_craftax_pixels(state, size)
        processed = jnp.clip(rendered / 255, 0, 1) * 255
        return processed.astype(jnp.uint8)

    render_and_process_vmap = jax.vmap(render_and_process, in_axes=(0, None))
    render_and_process_vmap_jit = jax.jit(render_and_process_vmap, static_argnums=(1,))

    is_sleeping_vmap = jax.vmap(is_sleeping)
    is_sleeping_vmap_jit = jax.jit(is_sleeping_vmap)

    is_null_do_crafting_vmap = jax.vmap(is_null_do_crafting)
    is_null_do_crafting_vmap_jit = jax.jit(is_null_do_crafting_vmap)

    is_null_do_action_vmap = jax.vmap(is_null_do_action)
    is_null_do_action_vmap_jit = jax.jit(is_null_do_action_vmap)

    is_null_place_block_vmap = jax.vmap(is_null_place_block)
    is_null_place_block_vmap_jit = jax.jit(is_null_place_block_vmap)

    is_move_player_vmap = jax.vmap(is_move_player)
    is_move_player_vmap_jit = jax.jit(is_move_player_vmap)

    is_sleeping_action_vmap = jax.vmap(is_sleeping_action)
    is_sleeping_action_vmap_jit = jax.jit(is_sleeping_action_vmap)

    pkl_dir = os.path.join(args.root, "state")
    action_dir = os.path.join(args.root, "action")
    frame_dir = os.path.join(args.root, "video")
    len_dir = os.path.join(args.root, "len")

    os.makedirs(action_dir, exist_ok=True)
    os.makedirs(frame_dir, exist_ok=True)
    os.makedirs(len_dir, exist_ok=True)

    pkl_files = sorted(glob.glob(os.path.join(pkl_dir, "*.npz")))

    start_episode = args.start_ep
    n_episodes = args.n_episodes
    pkl_files = pkl_files[start_episode:start_episode+n_episodes]

    CELL_SIZE = 16
    # CELL_SIZE = 64

    @jax.jit
    def check_is_null_action(env_states_batch, actions_batch):
        is_sleeping_batch = is_sleeping_vmap_jit(env_states_batch, actions_batch)
        is_null_do_crafting_batch = is_null_do_crafting_vmap_jit(env_states_batch, actions_batch)
        is_null_do_action_batch = is_null_do_action_vmap_jit(env_states_batch, actions_batch)
        is_null_place_block_batch = is_null_place_block_vmap_jit(env_states_batch, actions_batch)
        is_move_player_batch = is_move_player_vmap_jit(env_states_batch, actions_batch)
        is_sleeping_action_batch = is_sleeping_action_vmap_jit(env_states_batch, actions_batch)

        is_null_action = jnp.logical_or(
            is_sleeping_batch,
            jnp.logical_and(
                jnp.logical_not(is_sleeping_action_batch),
                jnp.logical_and(
                    jnp.logical_not(is_move_player_batch),
                    jnp.logical_and(
                        is_null_do_crafting_batch,
                        jnp.logical_and(
                            is_null_do_action_batch,
                            is_null_place_block_batch,
                        )
                    )
                )
            )
        )
        return is_null_action

    for pkl_file in tqdm(pkl_files):
        base_filename = os.path.basename(pkl_file)
        file_idx = base_filename.split('_')[1][:-4]
        
        actions, env_states = load_episode(pkl_file)
        episode_len = env_states.map.shape[0]
        assert episode_len == len(actions) + 1, f"{episode_len} != {len(actions) + 1}"

        # Render frames using GPU
        width = CELL_SIZE * 9
        height = CELL_SIZE * 9

        writer = cv2.VideoWriter(
            os.path.join(frame_dir, f"video_{file_idx}.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            20,
            (width, height)
        )

        total_actions = []
        total_nulls = []
        total_sleepings = []
        B = 300
        for i in range(0, episode_len, B):
            end_idx = min(i + B, episode_len)
            env_states_batch = jax.tree.map(lambda x: x[i:end_idx], env_states)
            frames = render_and_process_vmap_jit(env_states_batch, CELL_SIZE)

            for frame in frames:
                frame = cv2.cvtColor(
                    np.array(frame),
                    cv2.COLOR_RGB2BGR
                )
                writer.write(frame)

            if i == episode_len - 1:
                continue

            action_end_idx = min(end_idx, episode_len - 1)
            env_states_batch = jax.tree.map(lambda x: x[i:action_end_idx], env_states)
            actions_batch = actions[i:action_end_idx] 

            is_null_action = check_is_null_action(env_states_batch, actions_batch)

            actions_batch = np.array(actions_batch)
            is_null_action = np.array(is_null_action)
            sleeping_batch = np.array(env_states_batch.is_sleeping)

            total_actions.append(actions_batch)
            total_nulls.append(is_null_action)
            total_sleepings.append(sleeping_batch)

        total_actions = np.concatenate(total_actions)
        total_nulls = np.concatenate(total_nulls)
        total_sleepings = np.concatenate(total_sleepings)

        with open(os.path.join(action_dir, f"action_{file_idx}.pkl"), "wb") as f:
            pickle.dump((total_actions, total_nulls, total_sleepings), f)

        with open(os.path.join(len_dir, f"len_{file_idx}.txt"), "wt") as f:
            f.write(f"{len(actions)}\n")

        writer.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=str)
    parser.add_argument("--start_ep", type=int, default=0)
    parser.add_argument("--n_episodes", type=int, default=20000)
    args = parser.parse_args()
    main(args)
