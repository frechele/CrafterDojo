import argparse
import multiprocessing as mp
import os
from glob import glob

import numpy as np
import pandas as pd
from rich import print
from tqdm import tqdm


def score_one(npz_path: str) -> tuple[int, int, int]:
    basename = os.path.basename(npz_path)
    episode_id = int(basename.removeprefix("episode_").removesuffix(".npz"))
    with np.load(npz_path) as d:
        achievements = d["achievements"]
    score = int(np.asarray(achievements[-1]).sum())
    length = int(achievements.shape[0])
    return episode_id, score, length


def build_manifest(dataset_root: str, num_workers: int) -> pd.DataFrame:
    state_root = os.path.join(dataset_root, "state")
    if not os.path.isdir(state_root):
        raise FileNotFoundError(f"state/ not found under {dataset_root}")

    npz_paths = sorted(glob(os.path.join(state_root, "episode_*.npz")))
    if not npz_paths:
        raise FileNotFoundError(f"no episode_*.npz under {state_root}")

    print(f"[bold]Scoring {len(npz_paths)} episodes from[/bold] {state_root}")
    rows: list[tuple[int, int, int]] = []
    with mp.Pool(num_workers) as pool:
        for row in tqdm(pool.imap_unordered(score_one, npz_paths, chunksize=16),
                        total=len(npz_paths)):
            rows.append(row)

    df = pd.DataFrame(rows, columns=["episode_id", "achievements", "length"])
    df = df.sort_values("episode_id").reset_index(drop=True)
    return df


def write_allowlist(df: pd.DataFrame, top_k_pct: float, output_path: str) -> int:
    n_keep = max(1, int(round(len(df) * top_k_pct / 100.0)))
    ranked = df.sort_values(
        by=["achievements", "length"], ascending=[False, False]
    ).head(n_keep)
    ids = ranked["episode_id"].sort_values().tolist()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        for eid in ids:
            f.write(f"{eid:05d}\n")
    return n_keep


def main(args):
    df = build_manifest(args.dataset_root, args.num_workers)

    manifest_path = os.path.join(args.dataset_root, "episode_scores.csv")
    df.to_csv(manifest_path, index=False)
    print(f"[green]Wrote manifest[/green]: {manifest_path}")
    print(df["achievements"].describe())
    print("achievements histogram:")
    print(df["achievements"].value_counts().sort_index().to_string())

    if args.output_allowlist:
        n_keep = write_allowlist(df, args.top_k_pct, args.output_allowlist)
        print(
            f"[green]Wrote allowlist[/green] "
            f"({n_keep}/{len(df)} = top-{args.top_k_pct}%): {args.output_allowlist}"
        )


def main_cli(argv=None):
    parser = argparse.ArgumentParser(
        description="Rank playdata episodes by achievement count and emit top-k allow-list."
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        required=True,
        help="Path to the raw dataset root containing state/episode_*.npz",
    )
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument(
        "--top-k-pct",
        type=float,
        default=25.0,
        help="Keep top-k percent of episodes (by achievements desc, length desc).",
    )
    parser.add_argument(
        "--output-allowlist",
        type=str,
        default=None,
        help="If set, write an allow-list text file of selected episode IDs (zero-padded, one per line).",
    )

    args = parser.parse_args(argv)
    main(args)


if __name__ == "__main__":
    main_cli()
