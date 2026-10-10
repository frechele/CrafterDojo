import argparse
import os
import pickle
from concurrent.futures import ThreadPoolExecutor

import cv2
import jax
import jax.numpy as jnp
import numpy as np
from craftax.craftax_classic.constants import Action
from craftax.craftax_classic.envs.craftax_state import EnvState, Inventory, Mobs
from craftax.craftax_classic.renderer import render_craftax_pixels
from tqdm import tqdm

from dataset_generation.expert_behavior import rules as R


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
                        ),
                    ),
                ),
            ),
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
        ),
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
                    ),
                ),
            ),
        ),
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
            ),
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
                ),
            ),
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
            ),
        ),
    )


def is_sleeping_action(state: EnvState, action):
    return R.is_sleep(state, action)


def load_episode(npz_file: str):
    episode = np.load(npz_file)
    episode = {k: np.array(v) for k, v in episode.items()}

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


def write_video_cv2(path, frames_rgb, fps=20):
    """Match the legacy CrafterDojo dataset pipeline exactly."""
    height, width = frames_rgb.shape[1], frames_rgb.shape[2]
    writer = cv2.VideoWriter(
        path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    for frame in frames_rgb:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def main(args):
    @jax.jit
    def render_and_process_batch(env_states_batch):
        def _render_one(state):
            rendered = render_craftax_pixels(state, CELL_SIZE)
            processed = jnp.clip(rendered / 255, 0, 1) * 255
            return processed.astype(jnp.uint8)
        return jax.vmap(_render_one)(env_states_batch)

    @jax.jit
    def check_is_null_action(env_states_batch, actions_batch):
        is_sleeping_batch = jax.vmap(is_sleeping)(env_states_batch, actions_batch)
        is_null_do_crafting_batch = jax.vmap(is_null_do_crafting)(
            env_states_batch, actions_batch
        )
        is_null_do_action_batch = jax.vmap(is_null_do_action)(
            env_states_batch, actions_batch
        )
        is_null_place_block_batch = jax.vmap(is_null_place_block)(
            env_states_batch, actions_batch
        )
        is_move_player_batch = jax.vmap(is_move_player)(
            env_states_batch, actions_batch
        )
        is_sleeping_action_batch = jax.vmap(is_sleeping_action)(
            env_states_batch, actions_batch
        )

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
                        ),
                    ),
                ),
            ),
        )
        return is_null_action

    pkl_dir = os.path.join(args.root, "state")
    action_dir = os.path.join(args.root, "action")
    frame_dir = os.path.join(args.root, "video")
    len_dir = os.path.join(args.root, "len")

    os.makedirs(action_dir, exist_ok=True)
    os.makedirs(frame_dir, exist_ok=True)
    os.makedirs(len_dir, exist_ok=True)

    start_episode = args.start_ep
    n_episodes = args.n_episodes

    pkl_files = [
        os.path.join(pkl_dir, f"episode_{i:05d}.npz")
        for i in range(start_episode, start_episode + n_episodes)
    ]

    CELL_SIZE = 16
    RENDER_BATCH = 2000

    def save_outputs(file_idx, all_frames, actions, is_null, sleeping):
        """CPU-bound: write video + metadata (runs in background thread)."""
        write_video_cv2(
            os.path.join(frame_dir, f"video_{file_idx}.mp4"),
            all_frames,
        )
        with open(os.path.join(action_dir, f"action_{file_idx}.pkl"), "wb") as f:
            pickle.dump((actions, is_null, sleeping), f)
        with open(os.path.join(len_dir, f"len_{file_idx}.txt"), "wt") as f:
            f.write(f"{len(actions)}\n")

    SAVE_WORKERS = args.save_workers
    LOAD_WORKERS = args.load_workers

    save_executor = ThreadPoolExecutor(max_workers=SAVE_WORKERS)
    load_executor = ThreadPoolExecutor(max_workers=LOAD_WORKERS)
    save_futures = []

    # Prefetch multiple episodes using a bounded deque
    from collections import deque

    load_queue = deque()
    prefetch_ahead = LOAD_WORKERS
    for i in range(min(prefetch_ahead, len(pkl_files))):
        load_queue.append(load_executor.submit(load_episode, pkl_files[i]))

    for ep_i in tqdm(range(len(pkl_files)), desc="Processing episodes"):
        # Get current episode from prefetched queue
        actions, env_states = load_queue.popleft().result()

        # Submit next prefetch
        next_prefetch = ep_i + prefetch_ahead
        if next_prefetch < len(pkl_files):
            load_queue.append(
                load_executor.submit(load_episode, pkl_files[next_prefetch])
            )

        base_filename = os.path.basename(pkl_files[ep_i])
        file_idx = base_filename.split("_")[1][:-4]

        episode_len = env_states.map.shape[0]
        assert episode_len == len(actions) + 1, f"{episode_len} != {len(actions) + 1}"

        # Pad states to multiple of RENDER_BATCH to avoid JIT recompilation
        pad_len = (RENDER_BATCH - episode_len % RENDER_BATCH) % RENDER_BATCH
        if pad_len > 0:
            padded_states = jax.tree.map(
                lambda x: jnp.pad(x, [(0, pad_len)] + [(0, 0)] * (x.ndim - 1)),
                env_states,
            )
        else:
            padded_states = env_states

        # GPU: render all frames in fixed-size batches (no recompilation)
        all_frames = []
        for i in range(0, episode_len + pad_len, RENDER_BATCH):
            batch = jax.tree.map(lambda x: x[i:i + RENDER_BATCH], padded_states)
            frames = render_and_process_batch(batch)
            all_frames.append(frames)

        # GPU: null action check — pad to fixed size
        action_len = len(actions)
        action_pad = (RENDER_BATCH - action_len % RENDER_BATCH) % RENDER_BATCH
        action_states = jax.tree.map(lambda x: x[:-1], env_states)
        if action_pad > 0:
            padded_action_states = jax.tree.map(
                lambda x: jnp.pad(x, [(0, action_pad)] + [(0, 0)] * (x.ndim - 1)),
                action_states,
            )
            padded_actions = jnp.pad(actions, [(0, action_pad)] + [(0, 0)] * (actions.ndim - 1))
        else:
            padded_action_states = action_states
            padded_actions = actions
        is_null_padded = check_is_null_action(padded_action_states, padded_actions)

        # Transfer GPU→CPU and strip padding
        all_frames = np.concatenate(jax.device_get(all_frames), axis=0)[:episode_len]
        is_null = np.array(is_null_padded)[:action_len]
        sleeping = np.array(action_states.is_sleeping)
        actions_np = np.array(actions)

        # Free GPU memory
        del env_states, padded_states, action_states
        del padded_action_states, padded_actions, is_null_padded

        # Drain completed save futures to bound memory usage
        save_futures = [f for f in save_futures if not f.done()]
        while len(save_futures) >= SAVE_WORKERS:
            save_futures[0].result()
            save_futures.pop(0)

        # CPU: save outputs in background thread
        save_futures.append(
            save_executor.submit(
                save_outputs, file_idx, all_frames, actions_np, is_null, sleeping
            )
        )

    # Wait for all saves
    for f in save_futures:
        f.result()
    save_executor.shutdown()
    load_executor.shutdown()


def main_cli(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        default="data/episodes/Craftax-Classic-Symbolic-v1",
    )
    parser.add_argument("--start_ep", type=int, default=0)
    parser.add_argument("--n_episodes", type=int, default=20000)
    parser.add_argument("--save_workers", type=int, default=4)
    parser.add_argument("--load_workers", type=int, default=4)
    args = parser.parse_args(argv)
    main(args)


if __name__ == "__main__":
    main_cli()
