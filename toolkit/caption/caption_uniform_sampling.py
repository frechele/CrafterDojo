import argparse
import glob
import os
import pickle
import numpy as np
from tqdm import tqdm


def main(args):
    caption_path = args.root

    counters = {}
    for stat_file in args.stats:
        with open(stat_file, "rb") as f:
            local_stat = pickle.load(f)
        
        for k, v in local_stat.items():
            counters[k] = counters.get(k, 0) + v

    with open(os.path.join(caption_path, "caption_stats.pkl"), "wb") as f:
        pickle.dump(counters, f)

    total_caption = sum(counters.values())
    min_captions = min(counters.values())
    accept_rate = { k: min_captions / v for k, v in counters.items() }

    np.random.seed(args.seed)
    print("[Caption Stats]")
    print(counters)
    print(f"Total Caption: {int(total_caption):,}")

    caption_files = glob.glob(os.path.join(caption_path, "episode_*.pkl"))

    out_path = os.path.join(caption_path, "sampled")
    os.makedirs(out_path, exist_ok=True)

    new_counters = {}
    for caption_file in tqdm(caption_files):
        with open(caption_file, "rb") as f:
            captions = pickle.load(f)

        sampled = []
        for caption in captions:
            caption_type = caption[2]

            if np.random.rand() < accept_rate[caption_type]:
                sampled.append(caption)
                new_counters[caption_type] = new_counters.get(caption_type, 0) + 1

        basename = os.path.basename(caption_file) 
        with open(os.path.join(out_path, basename), "wb") as f:
            pickle.dump(sampled, f)
    print()

    new_total_caption = sum(new_counters.values())
    print("[New Caption Stats]")
    print(new_counters)
    print(f"New Total Caption: {int(new_total_caption):,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=str)
    parser.add_argument("stats", type=str, nargs="+")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
