import argparse
import glob
import os
import pickle

import ray
from tqdm import tqdm


def main(args):
    caption_files = glob.glob(os.path.join(args.root, "episode_*.pkl"))

    ray.init()

    @ray.remote
    def worker(filename):
        with open(filename, "rb") as f:
            captions = pickle.load(f)

        caption_set = set()
        for (_, _), caption, _ in captions:
            caption_set.add(caption)

        return caption_set

    workers = [worker.remote(filename) for filename in caption_files]
    total_captions = set()

    with tqdm(total=len(workers)) as pbar:
        while workers:
            done, workers = ray.wait(workers)
            for caption_set in ray.get(done):
                total_captions.update(caption_set)
            pbar.update(len(done))

    output_file = os.path.join(args.root, "all_captions.pkl")
    with open(output_file, "wb") as f:
        pickle.dump(total_captions, f)

    ray.shutdown()


def main_cli(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        default="data/episodes/Craftax-Classic-Symbolic-v1/caption",
    )
    args = parser.parse_args(argv)
    main(args)


if __name__ == "__main__":
    main_cli()
