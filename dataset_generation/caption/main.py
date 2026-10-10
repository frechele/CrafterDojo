import argparse
import glob
import os
import pickle
import random

import numpy as np
import ray
from tqdm import tqdm

from dataset_generation.caption.common import WINDOW_SIZE, EnvState, Inventory, Mobs
from dataset_generation.caption.engine import Engine, generate_caption_and_pivot
from dataset_generation.caption.rules import RULES


def load_episode(npz_file: str):
    with np.load(npz_file) as npz:
        ep = {k: npz[k] for k in npz.files}

    actions = ep["action"]
    ep_len = ep["map"].shape[0]

    env_states = [
        EnvState(
            map=ep["map"][step],
            mob_map=ep["mob_map"][step],
            player_position=ep["player_position"][step],
            player_direction=ep["player_direction"][step],
            player_health=ep["player_health"][step],
            player_food=ep["player_food"][step],
            player_drink=ep["player_drink"][step],
            player_energy=ep["player_energy"][step],
            is_sleeping=ep["is_sleeping"][step],
            player_recover=ep["player_recover"][step],
            player_hunger=ep["player_hunger"][step],
            player_thirst=ep["player_thirst"][step],
            player_fatigue=ep["player_fatigue"][step],
            inventory=Inventory(
                wood=ep["inventory_wood"][step],
                stone=ep["inventory_stone"][step],
                coal=ep["inventory_coal"][step],
                iron=ep["inventory_iron"][step],
                diamond=ep["inventory_diamond"][step],
                sapling=ep["inventory_sapling"][step],
                wood_pickaxe=ep["inventory_wood_pickaxe"][step],
                stone_pickaxe=ep["inventory_stone_pickaxe"][step],
                iron_pickaxe=ep["inventory_iron_pickaxe"][step],
                wood_sword=ep["inventory_wood_sword"][step],
                stone_sword=ep["inventory_stone_sword"][step],
                iron_sword=ep["inventory_iron_sword"][step],
            ),
            zombies=Mobs(
                position=ep["zombies_position"][step],
                health=ep["zombies_health"][step],
                mask=ep["zombies_mask"][step],
                attack_cooldown=ep["zombies_attack_cooldown"][step],
            ),
            cows=Mobs(
                position=ep["cows_position"][step],
                health=ep["cows_health"][step],
                mask=ep["cows_mask"][step],
                attack_cooldown=ep["cows_attack_cooldown"][step],
            ),
            skeletons=Mobs(
                position=ep["skeletons_position"][step],
                health=ep["skeletons_health"][step],
                mask=ep["skeletons_mask"][step],
                attack_cooldown=ep["skeletons_attack_cooldown"][step],
            ),
            arrows=Mobs(
                position=ep["arrows_position"][step],
                health=ep["arrows_health"][step],
                mask=ep["arrows_mask"][step],
                attack_cooldown=ep["arrows_attack_cooldown"][step],
            ),
            arrow_directions=ep["arrow_directions"][step],
            growing_plants_positions=ep["growing_plants_positions"][step],
            growing_plants_age=ep["growing_plants_age"][step],
            growing_plants_mask=ep["growing_plants_mask"][step],
            light_level=ep["light_level"][step],
            achievements=ep["achievements"][step],
            state_rng=ep["state_rng"][step],
            timestep=ep["timestep"][step],
        )
        for step in range(ep_len)
    ]

    return actions, env_states


def main(args):
    state_root = os.path.join(args.root, "state")
    caption_root = os.path.join(args.root, "caption")
    if args.dense:
        caption_root += "_dense"
    os.makedirs(caption_root, exist_ok=True)

    state_files = sorted(glob.glob(os.path.join(state_root, "*.npz")))
    state_files = state_files[args.start_ep : args.start_ep + args.n_episodes]

    ray.init()

    @ray.remote
    def worker(filename, dense):
        engine = Engine(RULES)

        episode_id = int(filename.split("_")[-1].split(".")[0])
        random.seed(episode_id)

        out_path = os.path.join(caption_root, os.path.basename(filename)[:-4] + ".pkl")
        actions, states = load_episode(filename)

        captions = []
        ep_len = len(states)
        window_size = WINDOW_SIZE + 1

        seq_stats = {
            "total_seqs": 0,
            "seq_counts": {1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
            "rule_combos": {},
        }

        for pivot, caption, name in generate_caption_and_pivot(engine, states, actions, dense):
            pivot = pivot + 1

            if pivot < window_size or pivot >= ep_len:
                continue

            start_idx = pivot - window_size
            end_idx = pivot
            captions.append(((start_idx, end_idx), caption, name))

            if dense:
                seq_stats["total_seqs"] += 1

                seq_len = len(caption)
                if seq_len in seq_stats["seq_counts"]:
                    seq_stats["seq_counts"][seq_len] += 1

                key = tuple(caption)
                if key in seq_stats["rule_combos"]:
                    seq_stats["rule_combos"][key] += 1
                else:
                    seq_stats["rule_combos"][key] = 1

        with open(out_path, "wb") as f:
            pickle.dump(captions, f)

        return engine.rule_statistics, seq_stats

    workers = [worker.remote(filename, args.dense) for filename in state_files]

    statistics = {}
    seq_stats = {
        "total_seqs": 0,
        "seq_counts": {1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
        "combo_num": 0,
        "rule_combos": {},
    }

    with tqdm(total=len(state_files)) as pbar:
        while workers:
            done, workers = ray.wait(workers)

            for stat, seq_stat in ray.get(done):
                for k, v in stat.items():
                    statistics[k] = statistics.get(k, 0) + v

                seq_stats["total_seqs"] += seq_stat["total_seqs"]

                for key, count in seq_stat["seq_counts"].items():
                    seq_stats["seq_counts"][key] += count

                for rule, count in seq_stat["rule_combos"].items():
                    seq_stats["rule_combos"][rule] = seq_stats["rule_combos"].get(rule, 0) + count

            pbar.update(len(done))

    stat_filename = f"statistics_{args.start_ep}_{args.start_ep + args.n_episodes}.pkl"
    with open(stat_filename, "wb") as f:
        pickle.dump(statistics, f)

    seq_stat_filename = f"seq_stats_{args.start_ep}_{args.start_ep + args.n_episodes}.pkl"
    with open(seq_stat_filename, "wb") as f:
        pickle.dump(seq_stats, f)

    ray.shutdown()


def main_cli(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        default="data/episodes/Craftax-Classic-Symbolic-v1",
    )
    parser.add_argument("--start_ep", type=int, default=0)
    parser.add_argument("--n_episodes", type=int, default=20000)
    parser.add_argument("--dense", action="store_true")
    args = parser.parse_args(argv)
    main(args)


if __name__ == "__main__":
    main_cli()
