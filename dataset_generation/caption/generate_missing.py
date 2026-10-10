"""One-off: generate captions for specific episode IDs (given gap 100000-109089)."""
import argparse
import glob
import os
import pickle
import random

import ray
from tqdm import tqdm

from dataset_generation.caption.common import WINDOW_SIZE
from dataset_generation.caption.engine import Engine, generate_caption_and_pivot
from dataset_generation.caption.main import load_episode
from dataset_generation.caption.rules import RULES


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--start_id", type=int, required=True)
    p.add_argument("--end_id", type=int, required=True, help="exclusive")
    p.add_argument("--skip_existing", action="store_true")
    args = p.parse_args()

    state_root = os.path.join(args.root, "state")
    caption_root = os.path.join(args.root, "caption")
    os.makedirs(caption_root, exist_ok=True)

    todo = []
    for ep_id in range(args.start_id, args.end_id):
        # state files use variable-width naming (5-digit for 0-99999, 6-digit for 100000+)
        for pattern in (f"episode_{ep_id:05d}.npz", f"episode_{ep_id:06d}.npz"):
            path = os.path.join(state_root, pattern)
            if os.path.exists(path):
                out = os.path.join(caption_root, os.path.basename(path)[:-4] + ".pkl")
                if args.skip_existing and os.path.exists(out):
                    break
                todo.append(path)
                break
    print(f"[missing-cap] {len(todo)} episodes to caption in [{args.start_id}, {args.end_id})")

    if not todo:
        return

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

        for pivot, caption, name in generate_caption_and_pivot(engine, states, actions, False):
            pivot = pivot + 1
            if pivot < window_size or pivot >= ep_len:
                continue
            start_idx = pivot - window_size
            end_idx = pivot
            captions.append(((start_idx, end_idx), caption, name))

        with open(out_path, "wb") as f:
            pickle.dump(captions, f)
        return True

    workers = [worker.remote(f) for f in todo]
    with tqdm(total=len(workers)) as pbar:
        while workers:
            done, workers = ray.wait(workers)
            ray.get(done)
            pbar.update(len(done))

    ray.shutdown()
    print("[missing-cap] done")


if __name__ == "__main__":
    main()
