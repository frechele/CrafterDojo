"""Compute release statistics for the CrafterPlay / CrafterCaption datasets.

Reads the local raw playdata trees (caption/ and len/ fields only — cheap) and
writes a JSON summary per dataset that docs/dataset/DATASET.md is built from.

Usage:
    python scripts/dataset_stats.py \
        --root <playdata_root>/gatherer-10B/Craftax-Classic-Symbolic-v1 \
        --name gatherer-10B \
        --out docs/dataset/stats/gatherer-10B.json
"""

import argparse
import json
import pickle
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


def _episode_stats(args):
    caption_path, len_path = args
    with open(caption_path, "rb") as f:
        captions = pickle.load(f)
    ep_len = int(Path(len_path).read_text())
    rule_counts = Counter(rule for (_, _, rule) in captions)
    text_counts = Counter(text for (_, text, _) in captions)
    seg_lengths = [end - start for ((start, end), _, _) in captions]
    return {
        "ep_len": ep_len,
        "n_captions": len(captions),
        "rule_counts": rule_counts,
        "text_counts": text_counts,
        "seg_len_sum": sum(seg_lengths),
        "seg_len_max": max(seg_lengths) if seg_lengths else 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Dataset root containing caption/ and len/")
    parser.add_argument("--name", required=True, help="Dataset name used in the output JSON")
    parser.add_argument("--out", required=True, help="Output JSON path")
    parser.add_argument("--n_episodes", type=int, default=None, help="Limit episodes (default: all)")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    root = Path(args.root)
    caption_files = sorted((root / "caption").glob("episode_*.pkl"))
    if args.n_episodes:
        caption_files = caption_files[: args.n_episodes]

    jobs = []
    for cf in caption_files:
        ep_id = cf.stem.split("_")[1]
        lf = root / "len" / f"len_{ep_id}.txt"
        if lf.exists():
            jobs.append((cf, lf))

    rule_counts = Counter()
    text_counts = Counter()
    ep_lens = []
    n_captions = []
    seg_len_sum = 0
    seg_len_max = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, res in enumerate(pool.map(_episode_stats, jobs, chunksize=64)):
            ep_lens.append(res["ep_len"])
            n_captions.append(res["n_captions"])
            rule_counts.update(res["rule_counts"])
            text_counts.update(res["text_counts"])
            seg_len_sum += res["seg_len_sum"]
            seg_len_max = max(seg_len_max, res["seg_len_max"])
            if (i + 1) % 1000 == 0:
                print(f"  {i + 1}/{len(jobs)} episodes", flush=True)

    ep_lens = np.array(ep_lens)
    n_captions_arr = np.array(n_captions)
    total_captions = int(n_captions_arr.sum())

    summary = {
        "dataset": args.name,
        "episodes": len(jobs),
        "episode_length": {
            "min": int(ep_lens.min()),
            "max": int(ep_lens.max()),
            "mean": float(ep_lens.mean()),
            "total_frames": int(ep_lens.sum()),
        },
        "captions": {
            "total_segments": total_captions,
            "per_episode_mean": float(n_captions_arr.mean()),
            "per_episode_min": int(n_captions_arr.min()),
            "per_episode_max": int(n_captions_arr.max()),
            "unique_texts": len(text_counts),
            "mean_segment_length_frames": seg_len_sum / total_captions if total_captions else 0,
            "max_segment_length_frames": int(seg_len_max),
        },
        "rule_counts": dict(rule_counts.most_common()),
        "top_caption_texts": dict(text_counts.most_common(20)),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
