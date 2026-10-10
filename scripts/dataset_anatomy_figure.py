"""Render the episode anatomy (file alignment) figure for docs/dataset/DATASET.md.

Shows how the five per-episode files line up on the frame axis: T+1 video frames
and states, T actions between consecutive frames, and captions spanning frame
ranges. Uses real frames from the sample release.

Usage:
    CRAFTERDOJO_SAMPLE_ROOT=<sample_dir> python scripts/dataset_anatomy_figure.py
"""

import os
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import FancyBboxPatch

OUT_PATH = Path("docs/dataset/figures/episode_anatomy.png")
SAMPLE_ROOT = Path(os.environ.get("CRAFTERDOJO_SAMPLE_ROOT", "sample_data"))
EPISODE_DIR = SAMPLE_ROOT / "raw" / "gatherer-10B" / "Craftax-Classic-Symbolic-v1"

# Real caption from episode 00000: ((3, 9), 'place sapling on grass', 'Construct')
CAPTION_START, CAPTION_TEXT = 3, "place sapling on grass"

INK = "#0b0b0b"
GRAY = "#6f6f6f"
HAIRLINE = "#c3c2b7"
STAGE_FILL = "#f0efec"
BLUE_TINT = "#d9e9f9"
BLUE = "#2a78d6"

SLOT = 1.24                       # frame-to-frame spacing
X0 = 2.95                         # centre of frame 0
XS = [X0 + i * SLOT for i in range(5)]          # frames 0..4
X_ELL = X0 + 5 * SLOT - 0.30                    # ellipsis
X_T = X0 + 5 * SLOT + 0.72                      # frame T


def box(ax, xc, yc, w, h, text, fill="#ffffff", fontsize=10, math=True,
        edge=INK, lw=1.2, text_color=INK):
    ax.add_patch(FancyBboxPatch(
        (xc - w / 2, yc - h / 2), w, h,
        boxstyle="round,pad=0.03,rounding_size=0.08",
        facecolor=fill, edgecolor=edge, linewidth=lw,
    ))
    if text:
        ax.text(xc, yc, text, ha="center", va="center",
                fontsize=fontsize, color=text_color)


def row_label(ax, y, name):
    ax.text(2.12, y, name, ha="right", va="center",
            fontsize=15, color=INK, family="monospace", fontweight="bold")


def main():
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["mathtext.fontset"] = "cm"

    reader = imageio.get_reader(str(EPISODE_DIR / "video" / "video_00000.mp4"))
    frames = {i: reader.get_data(i) for i in range(5)}
    n_frames = reader.count_frames()
    frames["T"] = reader.get_data(n_frames - 1)
    reader.close()

    fig, ax = plt.subplots(figsize=(9.6, 4.15), facecolor="#ffffff")
    ax.set_xlim(1.15, 10.6)
    ax.set_ylim(0.45, 4.85)
    ax.axis("off")

    y_video, y_state, y_action, y_caption = 4.05, 2.62, 1.92, 0.95
    zoom = 0.36  # 144px frame -> just under one slot wide at dpi 200

    # short connectors between the frame labels and the state entries
    for x in XS + [X_T]:
        ax.plot([x, x], [y_state + 0.26, y_video - 0.95], color=HAIRLINE,
                linewidth=1.0, linestyle=(0, (1, 2)), zorder=0)

    # --- video row ---------------------------------------------------------
    row_label(ax, y_video - 0.10, "video")
    for i, x in enumerate(XS):
        ab = AnnotationBbox(OffsetImage(frames[i], zoom=zoom), (x, y_video),
                            frameon=True, pad=0.05,
                            bboxprops=dict(edgecolor=INK, linewidth=1.0))
        ax.add_artist(ab)
        ax.text(x, y_video - 0.75, f"$o_{i}$", ha="center", va="center",
                fontsize=11, color=INK)
    ax.text(X_ELL, y_video, "$\\cdots$", ha="center", va="center",
            fontsize=15, color=INK)
    ab = AnnotationBbox(OffsetImage(frames["T"], zoom=zoom), (X_T, y_video),
                        frameon=True, pad=0.05,
                        bboxprops=dict(edgecolor=INK, linewidth=1.0))
    ax.add_artist(ab)
    ax.text(X_T, y_video - 0.75, "$o_T$", ha="center", va="center",
            fontsize=11, color=INK)

    # --- state row: one entry per frame ------------------------------------
    row_label(ax, y_state, "state")
    for i, x in enumerate(XS):
        box(ax, x, y_state, 0.78, 0.42, f"$s_{i}$", fill=STAGE_FILL,
            edge=HAIRLINE, fontsize=10)
    ax.text(X_ELL, y_state, "$\\cdots$", ha="center", va="center",
            fontsize=15, color=INK)
    box(ax, X_T, y_state, 0.78, 0.42, "$s_T$", fill=STAGE_FILL,
        edge=HAIRLINE, fontsize=10)

    # --- action row: one entry between consecutive frames ------------------
    row_label(ax, y_action, "action")
    for i in range(4):
        box(ax, (XS[i] + XS[i + 1]) / 2, y_action, 0.78, 0.42, f"$a_{i}$",
            fontsize=10)
    ax.text(X_ELL - 0.42, y_action, "$\\cdots$", ha="center", va="center",
            fontsize=15, color=INK)
    box(ax, (XS[4] + X_T) / 2 + 0.42, y_action, 0.94, 0.42, "$a_{T-1}$",
        fontsize=10)

    # --- caption row: a segment spanning a frame range ---------------------
    row_label(ax, y_caption, "caption")
    x_start = XS[CAPTION_START] - 0.45
    x_end = XS[4] + SLOT * 0.80          # runs past the visible frames
    ax.add_patch(FancyBboxPatch(
        (x_start, y_caption - 0.26), x_end - x_start, 0.52,
        boxstyle="round,pad=0.03,rounding_size=0.10",
        facecolor=BLUE_TINT, edgecolor=BLUE, linewidth=1.4,
    ))
    ax.text((x_start + x_end) / 2, y_caption,
            f'(({CAPTION_START}, 9),  “{CAPTION_TEXT}”)',
            ha="center", va="center", fontsize=9.5, color=INK)

    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=200, facecolor="#ffffff")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
