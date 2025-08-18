import argparse
import ray
import cv2
import glob
import pickle
import os
from rich import print
from tqdm import tqdm


def find_consecutive_null_segments(is_null_list, is_sleeping_list, min_length):
    if min_length <= 0:
        return [not is_null and not is_sleeping for is_null, is_sleeping in zip(is_null_list, is_sleeping_list)]
    
    keep_frames = []
    i = 0
    n = len(is_null_list)
    
    while i < n:
        if is_sleeping_list[i]:
            keep_frames.append(False)
            i += 1
        elif not is_null_list[i]:
            keep_frames.append(True)
            i += 1
        else:
            start = i
            while i < n and is_null_list[i]:
                i += 1
            null_length = i - start
            
            keep_segment = null_length >= min_length
            keep_frames.extend([keep_segment] * null_length)
    
    return keep_frames


def main(args):
    orig_action_root = os.path.join(args.root, "action")
    orig_video_root = os.path.join(args.root, "video")

    if args.filter > 0:
        output_path = os.path.join(os.path.basename(args.root), f"filtered_{args.filter}")
    else:
        output_path = os.path.join(os.path.basename(args.root), "filtered")

    output_path = os.path.join(os.path.dirname(args.root), output_path)
    os.makedirs(output_path, exist_ok=True)

    print(f"[bold]Output path[/bold]: {output_path}")

    out_action_root = os.path.join(output_path, "action")
    out_video_root = os.path.join(output_path, "video")
    out_len_root = os.path.join(output_path, "len")
    out_map_root = os.path.join(output_path, "map")
    os.makedirs(out_action_root, exist_ok=True)
    os.makedirs(out_video_root, exist_ok=True)
    os.makedirs(out_len_root, exist_ok=True)
    os.makedirs(out_map_root, exist_ok=True)

    print(f"[bold]Output Action Root[/bold]: {out_action_root}")
    print(f"[bold]Output Video Root[/bold]: {out_video_root}")
    print(f"[bold]Output Len Root[/bold]: {out_len_root}")      # Episode Length
    print(f"[bold]Output Mapping Root[/bold]: {out_map_root}")  # Index Mapping

    ray.init()

    @ray.remote
    def worker(episode_id):
        episode_str = f"{episode_id:05d}"
        action_path = os.path.join(orig_action_root, f"action_{episode_str}.pkl")
        video_path = os.path.join(orig_video_root, f"video_{episode_str}.mp4")

        out_action_path = os.path.join(out_action_root, f"action_{episode_str}.pkl")
        out_video_path = os.path.join(out_video_root, f"video_{episode_str}.mp4")
        out_len_path = os.path.join(out_len_root, f"len_{episode_str}.txt")
        out_map_path = os.path.join(out_map_root, f"map_{episode_str}.pkl")

        if os.path.exists(out_len_path):
            return

        with open(action_path, "rb") as f:
            orig_action = pickle.load(f)

        video = cv2.VideoCapture(video_path)
        vid_length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        assert vid_length == len(orig_action[0]) + 1, f"{vid_length} != {len(orig_action[0]) + 1}"

        actions, is_null_list, is_sleeping_list = orig_action
        keep_frames = find_consecutive_null_segments(is_null_list, is_sleeping_list, args.filter)

        index_map = []
        converted_idx = 0
        for i, keep in enumerate(keep_frames):
            if keep:
                index_map.append(converted_idx)
                converted_idx += 1
            else:
                index_map.append(-1)

        writer = cv2.VideoWriter(
            out_video_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            20,
            (width, height)
        )

        total_actions, total_nulls, total_sleepings = [], [], []
        
        for i, (action, is_null, is_sleeping, keep) in enumerate(zip(actions, is_null_list, is_sleeping_list, keep_frames)):
            ret, frame = video.read()
            assert ret

            if not keep:
                continue

            writer.write(frame)
            total_actions.append(action)
            total_nulls.append(is_null)
            total_sleepings.append(is_sleeping)

        ret, frame = video.read()
        if not ret:
            print("ERROR!!!!", vid_length, len(orig_action[0]), len(orig_action[1]), len(orig_action[2]))
        assert ret

        writer.write(frame)

        writer.release()
        video.release()

        with open(out_action_path, "wb") as f:
            pickle.dump((total_actions, total_nulls, total_sleepings), f)

        with open(out_len_path, "w") as f:
            f.write(f"{len(total_actions)}\n")
            
        with open(out_map_path, "wb") as f:
            pickle.dump(index_map, f)

    n_episodes = len(glob.glob(os.path.join(orig_action_root, "*.pkl")))

    workers = [worker.remote(i) for i in range(n_episodes)]

    with tqdm(total=n_episodes) as pbar:
        while workers:
            done, workers = ray.wait(workers)
            pbar.update(len(done))

    ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=str)
    parser.add_argument("--filter", type=int, default=20)
    
    args = parser.parse_args()
    main(args)
