import argparse
import os
import glob
import pickle
import numpy as np
import random
import ray
from tqdm import tqdm

from src.common import EnvState, Inventory, Mobs, WINDOW_SIZE
from src.engine import generate_caption_and_pivot, Engine
from src.rule import RULES


def load_episode(npz_file: str):
    episode = np.load(npz_file)
    episode = { k: np.array(v) for k, v in episode.items() }

    actions = episode["action"]

    env_states = []
    ep_len = episode["map"].shape[0]
    for step in range(ep_len):
        env_state = EnvState(
            map=np.array(episode["map"][step]),
            mob_map=np.array(episode["mob_map"][step]),

            player_position=np.array(episode["player_position"][step]),
            player_direction=np.array(episode["player_direction"][step]),
            player_health=np.array(episode["player_health"][step]),
            player_food=np.array(episode["player_food"][step]),
            player_drink=np.array(episode["player_drink"][step]),
            player_energy=np.array(episode["player_energy"][step]),
            is_sleeping=np.array(episode["is_sleeping"][step]),

            player_recover=np.array(episode["player_recover"][step]),
            player_hunger=np.array(episode["player_hunger"][step]),
            player_thirst=np.array(episode["player_thirst"][step]),
            player_fatigue=np.array(episode["player_fatigue"][step]),

            inventory=Inventory(
                wood=np.array(episode["inventory_wood"][step]),
                stone=np.array(episode["inventory_stone"][step]),
                coal=np.array(episode["inventory_coal"][step]),
                iron=np.array(episode["inventory_iron"][step]),
                diamond=np.array(episode["inventory_diamond"][step]),
                sapling=np.array(episode["inventory_sapling"][step]),
                wood_pickaxe=np.array(episode["inventory_wood_pickaxe"][step]),
                stone_pickaxe=np.array(episode["inventory_stone_pickaxe"][step]),
                iron_pickaxe=np.array(episode["inventory_iron_pickaxe"][step]),
                wood_sword=np.array(episode["inventory_wood_sword"][step]),
                stone_sword=np.array(episode["inventory_stone_sword"][step]),
                iron_sword=np.array(episode["inventory_iron_sword"][step]),
            ),

            zombies=Mobs(
                position=np.array(episode["zombies_position"][step]),
                health=np.array(episode["zombies_health"][step]),
                mask=np.array(episode["zombies_mask"][step]),
                attack_cooldown=np.array(episode["zombies_attack_cooldown"][step]),
            ),
            cows=Mobs(
                position=np.array(episode["cows_position"][step]),
                health=np.array(episode["cows_health"][step]),
                mask=np.array(episode["cows_mask"][step]),
                attack_cooldown=np.array(episode["cows_attack_cooldown"][step]),
            ),
            skeletons=Mobs(
                position=np.array(episode["skeletons_position"][step]),
                health=np.array(episode["skeletons_health"][step]),
                mask=np.array(episode["skeletons_mask"][step]),
                attack_cooldown=np.array(episode["skeletons_attack_cooldown"][step]),
            ),
            arrows=Mobs(
                position=np.array(episode["arrows_position"][step]),
                health=np.array(episode["arrows_health"][step]),
                mask=np.array(episode["arrows_mask"][step]),
                attack_cooldown=np.array(episode["arrows_attack_cooldown"][step]),
            ),
            arrow_directions=np.array(episode["arrow_directions"][step]),

            growing_plants_positions=np.array(episode["growing_plants_positions"][step]),
            growing_plants_age=np.array(episode["growing_plants_age"][step]),
            growing_plants_mask=np.array(episode["growing_plants_mask"][step]),

            light_level=np.array(episode["light_level"][step]),

            achievements=np.array(episode["achievements"][step]),

            state_rng=np.array(episode["state_rng"][step]),

            timestep=np.array(episode["timestep"][step]),
        )
        env_states.append(env_state)

    return actions, env_states


def main(args):
    state_root = os.path.join(args.root, "state")
    caption_root = os.path.join(args.root, "caption")
    os.makedirs(caption_root, exist_ok=True)

    state_files = sorted(glob.glob(os.path.join(state_root, "*.npz")))
    state_files = state_files[args.start_ep:args.start_ep+args.n_episodes]

    ray.init()

    @ray.remote
    def worker(filename):
        engine = Engine(RULES)

        episode_id = int(filename.split("_")[-1].split(".")[0])
        random.seed(episode_id)

        out_path = os.path.join(caption_root, os.path.basename(filename)[:-4] + ".pkl")
        actions, states = load_episode(filename)

        captions = []
        ep_len = len(states)
        window_size = WINDOW_SIZE + 1
        
        for pivot, caption, name in generate_caption_and_pivot(engine, states, actions):
            pivot = pivot + 1
            
            if pivot < window_size or pivot >= ep_len:
                continue

            start_idx = pivot - window_size
            end_idx = pivot
            captions.append(((start_idx, end_idx), caption, name))

        with open(out_path, "wb") as f:
            pickle.dump(captions, f)

        return engine.rule_statistics

    workers = [worker.remote(filename) for filename in state_files]

    statistics = {}
    
    with tqdm(total=len(state_files)) as pbar:
        while workers:
            done, workers = ray.wait(workers)

            for stat in ray.get(done):
                for k, v in stat.items():
                    statistics[k] = statistics.get(k, 0) + v

            pbar.update(len(done))

    stat_filename = f"statistics_{args.start_ep}_{args.start_ep+args.n_episodes}.pkl"
    with open(stat_filename, "wb") as f:
        pickle.dump(statistics, f)

    ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=str)
    parser.add_argument("--start_ep", type=int, default=0)
    parser.add_argument("--n_episodes", type=int, default=20000)
    args = parser.parse_args()
    main(args)
