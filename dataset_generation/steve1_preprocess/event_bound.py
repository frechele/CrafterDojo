import argparse
import ray
import pickle
import os
from tqdm import tqdm


def main(args):
    orig_action_root = os.path.join(args.root, "action")
    orig_caption_root = os.path.join(args.root, "caption")

    output_path = os.path.basename(args.root) + f"_{args.postfix}"
    output_path = os.path.join(os.path.dirname(args.root), output_path)

    out_event_root = os.path.join(output_path, "event")
    out_map_root = os.path.join(output_path, "map")
    os.makedirs(out_event_root, exist_ok=True)

    ray.init()

    @ray.remote
    def worker(episode_id):
        episode_str = f"{episode_id:05d}"
        action_path = os.path.join(orig_action_root, f"action_{episode_str}.pkl")
        caption_path = os.path.join(orig_caption_root, f"episode_{episode_str}.pkl")

        event_path = os.path.join(out_event_root, f"event_{episode_str}.pkl")
        map_path = os.path.join(out_map_root, f"map_{episode_str}.pkl")

        with open(action_path, "rb") as f:
            _, is_nulls, _ = pickle.load(f)

        with open(map_path, "rb") as f:
            index_map = pickle.load(f)
        index_map.append(len(index_map))  # the last frame is always kept

        assert len(index_map) == len(is_nulls) + 1

        with open(caption_path, "rb") as f:
            captions = pickle.load(f)

        events = {}
        for (_, pivot), caption, _ in captions:
            pivot = pivot - 1  # end_idx is inclusive.

            converted_pivot = index_map[pivot]
            if converted_pivot == -1:
                continue

            if converted_pivot not in events:
                events[converted_pivot] = []

            events[converted_pivot].append(caption)
        events = [ (k, set(v)) for k, v in events.items() ]
        events.sort(key=lambda x: x[0])

        event_timesteps = []
        last_event = events[0]
        event_timesteps.append(last_event[0])
        for_debugging = [0]
        for i, event in enumerate(events[1:], 1):
            if event[1] == last_event[1]:
                event_timesteps[-1] = event[0]
                for_debugging[-1] = i
                continue

            last_event = event
            event_timesteps.append(event[0])
            for_debugging.append(i)

        with open(event_path, "wb") as f:
            pickle.dump(event_timesteps, f)

    workers = [worker.remote(episode_id) for episode_id in range(args.start_ep, args.start_ep + args.n_episodes)]

    with tqdm(total=args.n_episodes) as pbar:
        while workers:
            done, workers = ray.wait(workers)
            pbar.update(len(done))

    ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="data/episodes/Craftax-Classic-Symbolic-v1")
    parser.add_argument("--postfix", type=str, default="filtered_20")
    parser.add_argument("--start_ep", type=int, default=0)
    parser.add_argument("--n_episodes", type=int, default=20000)

    args = parser.parse_args()
    main(args)
