"""Extract event-boundary timesteps from caption data.

Given episode captions (and optionally null-action filter maps), produces per-episode
pickle files listing the timesteps where the active caption set changes.

Two modes:
    filtered  -- applies a null-action index map so timesteps align with filtered data
    no_filter -- uses raw caption pivots directly (no index remapping)

Usage:
    uv run python -m crafterdojo.preprocess.event_bounding \
        --root data/episodes/Craftax-Classic-Symbolic-v1 \
        --postfix filtered_20 --n_episodes 20000

    uv run python -m crafterdojo.preprocess.event_bounding \
        --root data/episodes/Craftax-Classic-Symbolic-v1 \
        --no_filter --n_episodes 20000
"""

from __future__ import annotations

import argparse
import os
import pickle

import ray
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _consolidate_events(events: list[tuple[int, set[str]]]) -> list[int]:
    """Merge consecutive events with identical caption sets and return boundary timesteps."""
    if not events:
        return []

    event_timesteps: list[int] = []
    last_event = events[0]
    event_timesteps.append(last_event[0])

    for event in events[1:]:
        if event[1] == last_event[1]:
            # Same caption set -- update boundary to the later timestep
            event_timesteps[-1] = event[0]
        else:
            last_event = event
            event_timesteps.append(event[0])

    return event_timesteps


# ---------------------------------------------------------------------------
# Worker: filtered mode (uses action + map files)
# ---------------------------------------------------------------------------


@ray.remote
def _worker_filtered(
    episode_id: int,
    orig_action_root: str,
    orig_caption_root: str,
    out_event_root: str,
    out_map_root: str,
) -> None:
    episode_str = f"{episode_id:05d}"
    action_path = os.path.join(orig_action_root, f"action_{episode_str}.pkl")
    caption_path = os.path.join(orig_caption_root, f"episode_{episode_str}.pkl")
    event_path = os.path.join(out_event_root, f"event_{episode_str}.pkl")
    map_path = os.path.join(out_map_root, f"map_{episode_str}.pkl")

    with open(action_path, "rb") as f:
        _, is_nulls, _ = pickle.load(f)

    with open(map_path, "rb") as f:
        index_map = pickle.load(f)
    index_map.append(len(index_map))  # last frame is always kept

    assert len(index_map) == len(is_nulls) + 1

    with open(caption_path, "rb") as f:
        captions = pickle.load(f)

    events: dict[int, list[str]] = {}
    for (_, pivot), caption, _ in captions:
        pivot = pivot - 1  # end_idx is inclusive
        converted_pivot = index_map[pivot]
        if converted_pivot == -1:
            continue
        events.setdefault(converted_pivot, []).append(caption)

    sorted_events = [(k, set(v)) for k, v in events.items()]
    sorted_events.sort(key=lambda x: x[0])

    event_timesteps = _consolidate_events(sorted_events)

    with open(event_path, "wb") as f:
        pickle.dump(event_timesteps, f)


# ---------------------------------------------------------------------------
# Worker: no-filter mode (raw captions, no index remapping)
# ---------------------------------------------------------------------------


@ray.remote
def _worker_no_filter(
    episode_id: int,
    orig_caption_root: str,
    out_event_root: str,
) -> None:
    episode_str = f"{episode_id:05d}"
    caption_path = os.path.join(orig_caption_root, f"episode_{episode_str}.pkl")
    event_path = os.path.join(out_event_root, f"event_{episode_str}.pkl")

    with open(caption_path, "rb") as f:
        captions = pickle.load(f)

    events: dict[int, list[str]] = {}
    for (_, pivot), caption, _ in captions:
        pivot = pivot - 1  # end_idx is inclusive
        events.setdefault(pivot, []).append(caption)

    sorted_events = [(k, set(v)) for k, v in events.items()]
    sorted_events.sort(key=lambda x: x[0])

    event_timesteps = _consolidate_events(sorted_events)

    with open(event_path, "wb") as f:
        pickle.dump(event_timesteps, f)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args: argparse.Namespace) -> None:
    orig_caption_root = os.path.join(args.root, "caption")

    if args.no_filter:
        # no-filter mode: output goes alongside the source data
        output_path = args.root
    else:
        output_path = os.path.basename(args.root) + f"_{args.postfix}"
        output_path = os.path.join(os.path.dirname(args.root), output_path)

    out_event_root = os.path.join(output_path, "event")
    os.makedirs(out_event_root, exist_ok=True)

    ray.init()

    try:
        if args.no_filter:
            workers = [
                _worker_no_filter.remote(ep, orig_caption_root, out_event_root)
                for ep in range(args.start_ep, args.start_ep + args.n_episodes)
            ]
        else:
            orig_action_root = os.path.join(args.root, "action")
            out_map_root = os.path.join(output_path, "map")
            workers = [
                _worker_filtered.remote(
                    ep, orig_action_root, orig_caption_root, out_event_root, out_map_root
                )
                for ep in range(args.start_ep, args.start_ep + args.n_episodes)
            ]

        with tqdm(total=args.n_episodes) as pbar:
            while workers:
                done, workers = ray.wait(workers)
                pbar.update(len(done))
    finally:
        ray.shutdown()


def main_cli(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Extract event-boundary timesteps from captions")
    parser.add_argument("--root", type=str, required=True, help="Dataset root directory")
    parser.add_argument(
        "--postfix",
        type=str,
        default="filtered_20",
        help="Postfix for filtered output directory (ignored with --no_filter)",
    )
    parser.add_argument("--start_ep", type=int, default=0)
    parser.add_argument("--n_episodes", type=int, default=20000)
    parser.add_argument(
        "--no_filter",
        action="store_true",
        help="Skip null-action filtering (use raw caption pivots)",
    )

    args = parser.parse_args(argv)
    main(args)


if __name__ == "__main__":
    main_cli()
