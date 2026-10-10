"""Render the caption-rule distribution donut figure for docs/dataset/DATASET.md.

Reads the per-split JSON summaries produced by scripts/dataset_stats.py and writes
docs/dataset/figures/caption_rules.png.

Usage:
    python scripts/dataset_stats_figure.py
"""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

STATS_DIR = Path(os.environ.get("CRAFTERDOJO_STATS_DIR", "docs/dataset/stats"))
OUT_PATH = Path("docs/dataset/figures/caption_rules.png")

# Top-5 rules by combined segment count; the remaining 10 rules fold into "Other"
# (a donut stays glanceable at <= 6 segments; the full 15-rule table lives in the doc).
SEGMENTS = ["Explore", "Flee", "Harvest", "Stay", "Move Towards"]

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
# Categorical slots 1-5 (validated for the light surface); gray for "Other".
COLORS = {
    "Explore": "#2a78d6",
    "Flee": "#eb6834",
    "Harvest": "#1baf7a",
    "Stay": "#eda100",
    "Move Towards": "#e87ba4",
    "Other": "#c3c2b7",
}


def load_split(name):
    data = json.loads((STATS_DIR / f"{name}.json").read_text())
    counts = data["rule_counts"]
    total = sum(counts.values())
    values = [counts[r] for r in SEGMENTS]
    values.append(total - sum(values))  # Other
    return values, total


def draw_donut(ax, values, total, title):
    labels = SEGMENTS + ["Other"]
    shares = np.array(values) / total

    wedges, _ = ax.pie(
        values,
        startangle=90,
        counterclock=False,
        colors=[COLORS[r] for r in labels],
        wedgeprops=dict(width=0.40, edgecolor=SURFACE, linewidth=2),
    )

    # Percentage labels outside each wedge, in ink (never in the series color).
    for wedge, share in zip(wedges, shares):
        ang = np.deg2rad((wedge.theta1 + wedge.theta2) / 2)
        x, y = np.cos(ang), np.sin(ang)
        ax.text(
            1.16 * x, 1.16 * y, f"{share * 100:.1f}%",
            ha="center" if abs(x) < 0.35 else ("left" if x > 0 else "right"),
            va="center", fontsize=9, color=INK_SECONDARY,
        )

    ax.text(0, 0.10, title, ha="center", va="center",
            fontsize=13, fontweight="semibold", color=INK_PRIMARY)
    ax.text(0, -0.14, f"{total / 1e6:.1f}M segments", ha="center", va="center",
            fontsize=9.5, color=INK_SECONDARY)
    ax.set_aspect("equal")


def main():
    plt.rcParams["font.family"] = "sans-serif"

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.5), facecolor=SURFACE)
    for ax, (split, title) in zip(
        axes, [("survival-10B", "Survival"), ("gatherer-10B", "Gatherer")]
    ):
        values, total = load_split(split)
        draw_donut(ax, values, total, title)

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=COLORS[r], edgecolor="none")
        for r in SEGMENTS + ["Other"]
    ]
    fig.legend(
        handles, SEGMENTS + ["Other"],
        loc="lower center", ncol=6, frameon=False,
        bbox_to_anchor=(0.5, 0.015), fontsize=9.5,
        labelcolor=INK_SECONDARY, handlelength=1.0, handleheight=1.0,
        columnspacing=1.4, handletextpad=0.55,
    )

    fig.subplots_adjust(left=0.04, right=0.96, top=0.95, bottom=0.13, wspace=0.28)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=200, facecolor=SURFACE)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
