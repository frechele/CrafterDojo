import argparse
import glob
import os
import pickle
import statistics

import numpy as np
import ray
from tqdm import tqdm


def _caption_key(caption):
    if isinstance(caption, (list, tuple)):
        return tuple(caption)
    return caption


def _format_key(key):
    if isinstance(key, tuple):
        return " | ".join(key)
    return str(key)


def _resolve_target(counts, target):
    values = list(counts.values())
    if target == "median":
        return int(statistics.median(values))
    if target == "mean":
        return int(statistics.mean(values))
    if target == "min":
        return min(values)
    return int(target)


def _print_distribution(title, counts, top=None):
    items = sorted(counts.items(), key=lambda kv: -kv[1])
    total = sum(v for _, v in items)
    if top is not None:
        shown = items[:top]
    else:
        shown = items
    print(f"[{title}] unique={len(items):,} total={total:,}")
    for k, v in shown:
        pct = 100.0 * v / total if total else 0.0
        print(f"  {v:>10,}  {pct:5.2f}%  {_format_key(k)}")
    if top is not None and len(items) > top:
        print(f"  ... ({len(items) - top} more)")


def _count_main(args):
    caption_files = sorted(glob.glob(os.path.join(args.root, "episode_*.pkl")))
    if not caption_files:
        raise SystemExit(f"No episode_*.pkl under {args.root}")

    ray_init_kwargs = {}
    if args.num_cpus is not None:
        ray_init_kwargs["num_cpus"] = args.num_cpus
    ray.init(**ray_init_kwargs)

    @ray.remote
    def worker(filename):
        with open(filename, "rb") as f:
            captions = pickle.load(f)
        rule_counts = {}
        caption_counts = {}
        for _, caption, name in captions:
            rule_counts[name] = rule_counts.get(name, 0) + 1
            key = _caption_key(caption)
            caption_counts[key] = caption_counts.get(key, 0) + 1
        return rule_counts, caption_counts, len(captions)

    workers = [worker.remote(f) for f in caption_files]
    rule_counts = {}
    caption_counts = {}
    n_episodes = len(caption_files)
    total = 0

    with tqdm(total=len(workers), desc="count") as pbar:
        while workers:
            done, workers = ray.wait(workers, num_returns=min(128, len(workers)))
            for rc, cc, n in ray.get(done):
                for k, v in rc.items():
                    rule_counts[k] = rule_counts.get(k, 0) + v
                for k, v in cc.items():
                    caption_counts[k] = caption_counts.get(k, 0) + v
                total += n
            pbar.update(len(done))

    ray.shutdown()

    stats = {
        "rule_counts": rule_counts,
        "caption_counts": caption_counts,
        "n_episodes": n_episodes,
        "total_captions": total,
        "root": os.path.abspath(args.root),
    }

    out_path = args.stats_out or os.path.join(args.root, "balance_stats.pkl")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(stats, f)

    print(f"episodes: {n_episodes:,}")
    print(f"total captions: {total:,}")
    _print_distribution("rule_counts", rule_counts)
    _print_distribution("caption_counts", caption_counts, top=args.top)
    print(f"saved stats -> {out_path}")


def _apply_main(args):
    stats_path = args.stats or os.path.join(args.root, "balance_stats.pkl")
    with open(stats_path, "rb") as f:
        stats = pickle.load(f)

    if args.level == "rule":
        counts = dict(stats["rule_counts"])
    else:
        counts = dict(stats["caption_counts"])

    if args.include_rules:
        if args.level != "rule":
            raise SystemExit("--include_rules requires --level rule")
        keep = set(args.include_rules)
        counts = {k: v for k, v in counts.items() if k in keep}
        missing = keep - set(counts)
        if missing:
            raise SystemExit(f"include_rules not present in stats: {sorted(missing)}")

    if args.min_count > 0:
        kept = {k: v for k, v in counts.items() if v >= args.min_count}
        dropped = {k: v for k, v in counts.items() if v < args.min_count}
        counts = kept
    else:
        dropped = {}

    if not counts:
        raise SystemExit("No keys left after min_count filter")

    target = _resolve_target(counts, args.target)
    accept_rate = {k: min(1.0, target / v) for k, v in counts.items()}

    print(f"[apply] level={args.level} target={target:,} min_count={args.min_count} seed={args.seed}")
    print(f"  kept_keys={len(counts):,} dropped_keys={len(dropped):,}")
    if dropped:
        print("  dropped (below min_count):")
        for k, v in sorted(dropped.items(), key=lambda kv: -kv[1]):
            print(f"    {v:>8,}  {_format_key(k)}")

    rate_items = sorted(accept_rate.items(), key=lambda kv: counts[kv[0]], reverse=True)
    shown = rate_items if args.level == "rule" else rate_items[: args.top]
    print(f"  accept_rates (showing {len(shown)}/{len(rate_items)}):")
    for k, r in shown:
        print(f"    rate={r:6.4f}  count={counts[k]:>10,}  {_format_key(k)}")

    caption_files = sorted(glob.glob(os.path.join(args.root, "episode_*.pkl")))
    os.makedirs(args.out, exist_ok=True)

    ray_init_kwargs = {}
    if args.num_cpus is not None:
        ray_init_kwargs["num_cpus"] = args.num_cpus
    ray.init(**ray_init_kwargs)

    level = args.level
    seed = args.seed
    out_dir = args.out
    accept_rate_ref = ray.put(accept_rate)

    @ray.remote
    def worker(filename, idx, rates):
        with open(filename, "rb") as f:
            captions = pickle.load(f)
        rng = np.random.default_rng(seed + idx)
        local_rule = {}
        local_caption = {}
        kept = []
        for item in captions:
            _, caption, name = item
            key = name if level == "rule" else _caption_key(caption)
            rate = rates.get(key, 0.0)
            if rate <= 0.0:
                continue
            if rate >= 1.0 or rng.random() < rate:
                kept.append(item)
                local_rule[name] = local_rule.get(name, 0) + 1
                ckey = _caption_key(caption)
                local_caption[ckey] = local_caption.get(ckey, 0) + 1
        out_path = os.path.join(out_dir, os.path.basename(filename))
        with open(out_path, "wb") as f:
            pickle.dump(kept, f)
        return local_rule, local_caption, len(kept)

    workers_list = [worker.remote(f, i, accept_rate_ref) for i, f in enumerate(caption_files)]

    new_rule = {}
    new_caption = {}
    total_kept = 0
    with tqdm(total=len(workers_list), desc="apply") as pbar:
        while workers_list:
            done, workers_list = ray.wait(workers_list, num_returns=min(128, len(workers_list)))
            for lr, lc, n in ray.get(done):
                for k, v in lr.items():
                    new_rule[k] = new_rule.get(k, 0) + v
                for k, v in lc.items():
                    new_caption[k] = new_caption.get(k, 0) + v
                total_kept += n
            pbar.update(len(done))

    ray.shutdown()

    print(f"kept {total_kept:,} of {stats['total_captions']:,} "
          f"({100.0 * total_kept / max(1, stats['total_captions']):.2f}%)")
    _print_distribution("rule_counts (after)", new_rule)
    _print_distribution("caption_counts (after)", new_caption, top=args.top)
    print(f"wrote balanced captions -> {out_dir}")


def main_cli(argv=None):
    parser = argparse.ArgumentParser(description="Balance caption datasets (rule or caption-text level)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_count = sub.add_parser("count", help="aggregate caption distribution")
    p_count.add_argument("--root", type=str, required=True, help="caption dir with episode_*.pkl")
    p_count.add_argument("--stats_out", type=str, default=None)
    p_count.add_argument("--num_cpus", type=int, default=None)
    p_count.add_argument("--top", type=int, default=30)

    p_apply = sub.add_parser("apply", help="write balanced caption dir")
    p_apply.add_argument("--root", type=str, required=True, help="caption dir with episode_*.pkl")
    p_apply.add_argument("--out", type=str, required=True, help="output dir for balanced episode_*.pkl")
    p_apply.add_argument("--stats", type=str, default=None)
    p_apply.add_argument("--level", choices=["rule", "caption"], required=True)
    p_apply.add_argument("--target", type=str, default="median",
                         help="'median' | 'mean' | 'min' | integer")
    p_apply.add_argument("--min_count", type=int, default=0,
                         help="drop keys with count < min_count before balancing")
    p_apply.add_argument("--include_rules", type=lambda s: [r.strip() for r in s.split(",") if r.strip()],
                         default=None,
                         help="comma-separated rule names to keep (level=rule only); other rules dropped")
    p_apply.add_argument("--seed", type=int, default=42)
    p_apply.add_argument("--num_cpus", type=int, default=None)
    p_apply.add_argument("--top", type=int, default=30)

    args = parser.parse_args(argv)
    if args.cmd == "count":
        _count_main(args)
    else:
        _apply_main(args)


if __name__ == "__main__":
    main_cli()
