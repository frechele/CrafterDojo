"""Extract embeddings for BC / CVAE training.

Subcommands (dispatched via CLI):
    clip           -- per-timestep CrafterCLIP video embeddings (sliding window)
    clip-ach-text  -- per-timestep CrafterCLIP *text* embeddings (attn_achievement weights)
    sbert          -- per-timestep SentenceBERT text embeddings
    paired         -- paired (text, video) embeddings for CVAE prior training
    text           -- text+video embeddings from caption_v2 data

Usage:
    uv run python -m crafterdojo.preprocess.embedding clip \
        --root data/episodes/Craftax-Classic-Symbolic-v1_filtered_20 \
        --n_episodes 625

    uv run python -m crafterdojo.preprocess.embedding sbert \
        --root data/episodes/Craftax-Classic-Symbolic-v1 \
        --postfix filtered_20 --n_episodes 20000

    uv run python -m crafterdojo.preprocess.embedding clip-ach-text \
        --root data/episodes/Craftax-Classic-Symbolic-v1 \
        --postfix filtered_20 --n_episodes 20000 \
        --ckpt models/mineclip/attn_achievement.weights

    uv run python -m crafterdojo.preprocess.embedding paired \
        --root data/episodes/Craftax-Classic-Symbolic-v1 \
        --n_episodes 200 --ckpt models/mineclip/attn_achievement.weights \
        --output models/steve1/text_embeddings_ach.npz

    uv run python -m crafterdojo.preprocess.embedding text \
        --root data/episodes/Craftax-Classic-Symbolic-v1 \
        --n_episodes 1000
"""

from __future__ import annotations

import argparse
import os
import pickle

import cv2
import numpy as np
import torch
from tqdm import tqdm

from crafterdojo.data.vpt_dataset import Episode as VPTEpisode
from crafterdojo.data.steve1_dataset import Episode as Steve1Episode
from crafterdojo.model.crafterclip import load_mineclip_wconfig, CRAFTER_MINECLIP_CONFIG


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLIP_WINDOW_SIZE = 6
CLIP_EMBED_DIM = 512
NONE_EMBED_OFFSET = 5


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _image_preprocess(frame: np.ndarray) -> np.ndarray:
    """Resize to 144x144 and convert HWC -> CHW."""
    return cv2.resize(frame, (144, 144)).transpose(2, 0, 1)


def _reconstruct_event_captions(
    caption_path: str,
    map_path: str,
    action_path: str,
) -> list[tuple[int, str]]:
    """Reconstruct event-caption mapping from raw caption and map files.

    Returns list of (timestep, joined_caption_text) tuples sorted by timestep.
    Each entry marks a boundary where the caption set changes.
    """
    with open(action_path, "rb") as f:
        _, is_nulls, _ = pickle.load(f)

    with open(map_path, "rb") as f:
        index_map = pickle.load(f)
    index_map.append(len(index_map))

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

    if not sorted_events:
        return []

    # Consolidate consecutive events with identical caption sets
    consolidated: list[tuple[int, set[str]]] = []
    last_event = sorted_events[0]
    consolidated.append(last_event)
    for event in sorted_events[1:]:
        if event[1] == last_event[1]:
            consolidated[-1] = event
            last_event = event
        else:
            last_event = event
            consolidated.append(event)

    return [(t, ", ".join(sorted(caption_set))) for t, caption_set in consolidated]


def _create_per_timestep_embeddings(
    event_captions: list[tuple[int, str]],
    episode_len: int,
    caption_to_embed: dict[str, np.ndarray],
    embed_dim: int,
) -> np.ndarray:
    """Create per-timestep embedding array from event-caption mapping.

    For timesteps between event boundaries, assigns the caption embedding
    of the current segment.
    """
    embeddings = np.zeros((episode_len, embed_dim), dtype=np.float32)

    if not event_captions:
        return embeddings

    for i, (t, caption) in enumerate(event_captions):
        embed = caption_to_embed[caption]
        next_t = event_captions[i + 1][0] if i + 1 < len(event_captions) else episode_len
        embeddings[t:next_t] = embed

    # Fill timesteps before the first event with the first caption
    first_t = event_captions[0][0]
    if first_t > 0:
        embeddings[:first_t] = caption_to_embed[event_captions[0][1]]

    return embeddings


# ---------------------------------------------------------------------------
# Subcommand: clip  (per-timestep CrafterCLIP video embeddings)
# ---------------------------------------------------------------------------


@torch.no_grad()
def extract_clip(args: argparse.Namespace) -> None:
    """Extract per-timestep CrafterCLIP video embeddings using a sliding window."""
    embedding_root = os.path.join(args.root, "clip_embedding")
    os.makedirs(embedding_root, exist_ok=True)

    device = torch.device("cuda:0")
    mineclip = load_mineclip_wconfig(device)
    mineclip.eval()

    B = args.batch_size

    for episode_id in tqdm(range(args.start_ep, args.start_ep + args.n_episodes)):
        episode = VPTEpisode(args.root, episode_id)

        episode_embeddings = np.zeros((CLIP_WINDOW_SIZE - 1, CLIP_EMBED_DIM), dtype=np.float32)
        n_frames = len(episode) + 1
        for start_idx in tqdm(range(0, n_frames - CLIP_WINDOW_SIZE + 1, B)):
            real_batch = min(B, n_frames - start_idx - CLIP_WINDOW_SIZE + 1)

            end_idx = start_idx + (real_batch - 1) + CLIP_WINDOW_SIZE
            frames = episode.load_frames((start_idx, end_idx))[start_idx:end_idx]

            all_frames = np.array([_image_preprocess(frame) for frame in frames])
            indices = np.arange(real_batch)[:, None] + np.arange(CLIP_WINDOW_SIZE)[None, :]
            batch = all_frames[indices].astype(np.float32)
            batch = torch.from_numpy(batch).to(device)

            embeddings = mineclip.encode_video(batch).cpu().numpy()
            episode_embeddings = np.concatenate([episode_embeddings, embeddings], axis=0)

        print("episode length", len(episode))
        print("episode_embeddings.shape", episode_embeddings.shape)

        np.savez_compressed(
            os.path.join(embedding_root, f"{episode_id:05d}.npz"),
            window_size=CLIP_WINDOW_SIZE,
            embeddings=episode_embeddings,
        )


# ---------------------------------------------------------------------------
# Subcommand: sbert  (per-timestep SentenceBERT text embeddings)
# ---------------------------------------------------------------------------


def extract_sbert(args: argparse.Namespace) -> None:
    """Extract per-timestep SentenceBERT text embeddings for BC training."""
    from sentence_transformers import SentenceTransformer

    orig_caption_root = os.path.join(args.root, "caption")
    orig_action_root = os.path.join(args.root, "action")

    filtered_root = os.path.basename(args.root) + f"_{args.postfix}"
    filtered_root = os.path.join(os.path.dirname(args.root), filtered_root)

    map_root = os.path.join(filtered_root, "map")
    len_root = os.path.join(filtered_root, "len")

    embedding_root = os.path.join(filtered_root, "sbert_embedding")
    os.makedirs(embedding_root, exist_ok=True)

    # Load SentenceBERT model
    print(f"Loading SentenceBERT model: {args.model_name}")
    sbert = SentenceTransformer(args.model_name)
    embed_dim = sbert.get_sentence_embedding_dimension()
    print(f"Embedding dimension: {embed_dim}")

    # Pass 1: collect all unique captions
    print("Pass 1: Collecting unique captions...")
    all_unique_captions: set[str] = set()
    episode_event_captions: dict[int, list[tuple[int, str]]] = {}

    for episode_id in tqdm(range(args.start_ep, args.start_ep + args.n_episodes)):
        episode_str = f"{episode_id:05d}"
        caption_path = os.path.join(orig_caption_root, f"episode_{episode_str}.pkl")
        map_path = os.path.join(map_root, f"map_{episode_str}.pkl")
        action_path = os.path.join(orig_action_root, f"action_{episode_str}.pkl")

        if not all(os.path.exists(p) for p in [caption_path, map_path, action_path]):
            continue

        event_captions = _reconstruct_event_captions(caption_path, map_path, action_path)
        episode_event_captions[episode_id] = event_captions

        for _, caption in event_captions:
            all_unique_captions.add(caption)

    # Batch encode all unique captions
    print(f"Encoding {len(all_unique_captions)} unique captions with SentenceBERT...")
    unique_captions_list = sorted(all_unique_captions)
    unique_embeddings = sbert.encode(
        unique_captions_list, batch_size=args.batch_size, show_progress_bar=True
    )
    caption_to_embed = dict(zip(unique_captions_list, unique_embeddings))

    # Pass 2: create per-timestep embeddings
    print("Pass 2: Creating per-timestep embeddings...")
    for episode_id in tqdm(range(args.start_ep, args.start_ep + args.n_episodes)):
        if episode_id not in episode_event_captions:
            continue

        episode_str = f"{episode_id:05d}"
        len_path = os.path.join(len_root, f"len_{episode_str}.txt")

        with open(len_path, "rt") as f:
            episode_len = int(f.read())
        total_len = episode_len + 1  # T frames = T-1 actions + 1 initial

        event_captions = episode_event_captions[episode_id]
        embeddings = _create_per_timestep_embeddings(
            event_captions, total_len, caption_to_embed, embed_dim
        )

        np.savez_compressed(
            os.path.join(embedding_root, f"{episode_str}.npz"),
            embeddings=embeddings,
        )

    print(f"Done. Saved embeddings to {embedding_root}")


# ---------------------------------------------------------------------------
# Subcommand: clip-ach-text  (per-timestep MineCLIP text embeddings,
#                              attn_achievement weights)
# ---------------------------------------------------------------------------


@torch.no_grad()
def extract_clip_ach_text(args: argparse.Namespace) -> None:
    """Extract per-timestep CrafterCLIP text embeddings (attn_achievement weights)."""
    orig_caption_root = os.path.join(args.root, "caption")
    orig_action_root = os.path.join(args.root, "action")

    filtered_root = os.path.basename(args.root) + f"_{args.postfix}"
    filtered_root = os.path.join(os.path.dirname(args.root), filtered_root)

    map_root = os.path.join(filtered_root, "map")
    len_root = os.path.join(filtered_root, "len")

    embedding_root = os.path.join(filtered_root, "clip_ach_embedding")
    os.makedirs(embedding_root, exist_ok=True)

    # Load MineCLIP model
    device = torch.device("cuda:0")
    clip_config = CRAFTER_MINECLIP_CONFIG.copy()
    clip_config["ckpt"] = args.ckpt
    print(f"Loading MineCLIP from: {args.ckpt}")
    mineclip = load_mineclip_wconfig(device, cfg=clip_config)
    mineclip.eval()
    print(f"Embedding dimension: {CLIP_EMBED_DIM}")

    # Pass 1: collect all unique captions
    print("Pass 1: Collecting unique captions...")
    all_unique_captions: set[str] = set()
    episode_event_captions: dict[int, list[tuple[int, str]]] = {}

    for episode_id in tqdm(range(args.start_ep, args.start_ep + args.n_episodes)):
        episode_str = f"{episode_id:05d}"
        caption_path = os.path.join(orig_caption_root, f"episode_{episode_str}.pkl")
        map_path = os.path.join(map_root, f"map_{episode_str}.pkl")
        action_path = os.path.join(orig_action_root, f"action_{episode_str}.pkl")

        if not all(os.path.exists(p) for p in [caption_path, map_path, action_path]):
            continue

        event_captions = _reconstruct_event_captions(caption_path, map_path, action_path)
        episode_event_captions[episode_id] = event_captions

        for _, caption in event_captions:
            all_unique_captions.add(caption)

    # Batch encode all unique captions with MineCLIP text encoder
    print(f"Encoding {len(all_unique_captions)} unique captions with MineCLIP...")
    unique_captions_list = sorted(all_unique_captions)

    all_embeddings = []
    for batch_start in tqdm(range(0, len(unique_captions_list), args.batch_size)):
        batch_end = min(batch_start + args.batch_size, len(unique_captions_list))
        batch_captions = unique_captions_list[batch_start:batch_end]
        batch_embeddings = mineclip.encode_text(batch_captions).cpu().numpy()
        all_embeddings.append(batch_embeddings)

    all_embeddings_arr = np.concatenate(all_embeddings, axis=0)
    caption_to_embed = dict(zip(unique_captions_list, all_embeddings_arr))

    # Pass 2: create per-timestep embeddings
    print("Pass 2: Creating per-timestep embeddings...")
    for episode_id in tqdm(range(args.start_ep, args.start_ep + args.n_episodes)):
        if episode_id not in episode_event_captions:
            continue

        episode_str = f"{episode_id:05d}"
        out_path = os.path.join(embedding_root, f"{episode_str}.npz")
        if os.path.exists(out_path):
            continue

        len_path = os.path.join(len_root, f"len_{episode_str}.txt")

        with open(len_path, "rt") as f:
            episode_len = int(f.read())
        total_len = episode_len + 1

        event_captions = episode_event_captions[episode_id]
        embeddings = _create_per_timestep_embeddings(
            event_captions, total_len, caption_to_embed, CLIP_EMBED_DIM
        )

        np.savez(out_path, embeddings=embeddings)

    print(f"Done. Saved embeddings to {embedding_root}")


# ---------------------------------------------------------------------------
# Subcommand: paired  (paired text+video embeddings for CVAE prior)
# ---------------------------------------------------------------------------


@torch.no_grad()
def extract_paired(args: argparse.Namespace) -> None:
    """Extract paired (text, video) embeddings for CVAE prior training."""
    device = torch.device("cuda:0")
    clip_config = CRAFTER_MINECLIP_CONFIG.copy()
    clip_config["ckpt"] = args.ckpt
    print(f"Loading MineCLIP from: {args.ckpt}")
    mineclip = load_mineclip_wconfig(device, cfg=clip_config)
    mineclip.eval()

    all_vid_embeds: list[np.ndarray] = []
    all_captions: list[str] = []

    for episode_id in tqdm(range(args.start_ep, args.start_ep + args.n_episodes)):
        caption_path = os.path.join(args.root, "caption", f"episode_{episode_id:05d}.pkl")
        if not os.path.exists(caption_path):
            continue

        with open(caption_path, "rb") as f:
            captions = pickle.load(f)

        # Filter valid captions
        valid_captions = []
        for (start_t, end_t), caption_text, _ in captions:
            if end_t - start_t < CLIP_WINDOW_SIZE or end_t <= NONE_EMBED_OFFSET:
                continue
            valid_captions.append((start_t, end_t, caption_text))

        if not valid_captions:
            continue

        # Load all frames once
        episode = VPTEpisode(args.root, episode_id)
        all_frames = episode.load_frames()

        # Collect video windows for batch processing
        batch_windows: list[np.ndarray] = []
        batch_texts: list[str] = []
        for start_t, end_t, caption_text in valid_captions:
            frames = all_frames[start_t:end_t]
            if len(frames) < CLIP_WINDOW_SIZE:
                continue

            mid = len(frames) // 2
            win_start = max(0, mid - CLIP_WINDOW_SIZE // 2)
            win_end = win_start + CLIP_WINDOW_SIZE
            if win_end > len(frames):
                win_end = len(frames)
                win_start = win_end - CLIP_WINDOW_SIZE

            window = np.array([_image_preprocess(f) for f in frames[win_start:win_end]])
            batch_windows.append(window)
            batch_texts.append(caption_text)

        if not batch_windows:
            continue

        # Encode video in batches
        for i in range(0, len(batch_windows), args.vid_batch_size):
            batch = np.stack(batch_windows[i : i + args.vid_batch_size])
            video_input = torch.from_numpy(batch.astype(np.float32)).to(device)
            vid_embeds = mineclip.encode_video(video_input).cpu().numpy()
            all_vid_embeds.extend(vid_embeds)

        all_captions.extend(batch_texts)

    # Batch encode all text
    print(f"\nEncoding {len(all_captions)} captions...")
    text_embeds_list: list[np.ndarray] = []
    for i in tqdm(range(0, len(all_captions), args.text_batch_size)):
        batch = all_captions[i : i + args.text_batch_size]
        embeds = mineclip.encode_text(batch).cpu().numpy()
        text_embeds_list.append(embeds)

    all_text_embeds = np.concatenate(text_embeds_list, axis=0)
    all_vid_embeds_arr = np.array(all_vid_embeds)

    print(f"text_embeddings: {all_text_embeds.shape}")
    print(f"vid_embeddings: {all_vid_embeds_arr.shape}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.savez(
        args.output,
        text_embeddings=all_text_embeds,
        vid_embeddings=all_vid_embeds_arr,
        captions=np.array(all_captions),
    )
    print(f"Saved to {args.output}")


# ---------------------------------------------------------------------------
# Subcommand: text  (text+video embeddings from caption_v2)
# ---------------------------------------------------------------------------


@torch.no_grad()
def extract_text(args: argparse.Namespace) -> None:
    """Extract text+video embeddings from caption_v2 data."""
    device = torch.device("cuda:0")
    clip_config = CRAFTER_MINECLIP_CONFIG.copy()
    if args.ckpt is not None:
        clip_config["ckpt"] = args.ckpt
    mineclip = load_mineclip_wconfig(device, cfg=clip_config)
    mineclip.eval()

    B = args.batch_size

    results: dict[str, np.ndarray | None] = {
        "captions": None,
        "text_embeddings": None,
        "vid_embeddings": None,
    }

    for episode_id in tqdm(range(args.start_ep, args.start_ep + args.n_episodes)):
        episode = Steve1Episode(args.root, episode_id)

        caption_path = os.path.join(args.root, "caption_v2", f"episode_{episode_id:05d}.pkl")
        with open(caption_path, "rb") as f:
            captions = pickle.load(f)

        frames: list[np.ndarray] = []
        caption_texts: list[str] = []
        for caption in captions:
            start_timestep, end_timestep = caption[0]
            caption_text = caption[1]

            if end_timestep <= NONE_EMBED_OFFSET:
                continue

            frames.append(
                episode.load_frames(only_range=(start_timestep, end_timestep))[
                    start_timestep:end_timestep
                ]
            )
            caption_texts.append(caption_text)

        for batch_start in range(0, len(caption_texts), B):
            batch_end = min(batch_start + B, len(caption_texts))

            batch_texts = caption_texts[batch_start:batch_end]

            local_frames = np.stack(frames[batch_start:batch_end]).transpose(0, 1, 4, 2, 3)
            embeddings = (
                mineclip.encode_video(torch.from_numpy(local_frames).to(device)).cpu().numpy()
            )
            batch_embeddings = mineclip.encode_text(batch_texts).cpu().numpy()

            if results["text_embeddings"] is None:
                results["text_embeddings"] = batch_embeddings
                results["vid_embeddings"] = embeddings
                results["captions"] = np.array(batch_texts)
            else:
                results["text_embeddings"] = np.concatenate(
                    [results["text_embeddings"], batch_embeddings], axis=0
                )
                results["vid_embeddings"] = np.concatenate(
                    [results["vid_embeddings"], embeddings], axis=0
                )
                results["captions"] = np.concatenate(
                    [results["captions"], np.array(batch_texts)], axis=0
                )

    print({k: v.shape for k, v in results.items()})

    postfix = f"_{args.postfix}" if args.postfix else ""
    out_filename = f"text_embeddings{postfix}.npz"
    np.savez(os.path.join(args.root, out_filename), **results)


# ---------------------------------------------------------------------------
# CLI dispatch
# ---------------------------------------------------------------------------


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments shared by all subcommands."""
    parser.add_argument("--root", type=str, required=True, help="Dataset root directory")
    parser.add_argument("--start_ep", type=int, default=0)
    parser.add_argument("--n_episodes", type=int, default=20000)


def main_cli(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Extract embeddings for BC / CVAE training",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # -- clip --
    p_clip = subparsers.add_parser("clip", help="CrafterCLIP video embeddings (sliding window)")
    _add_common_args(p_clip)
    p_clip.add_argument("--batch_size", type=int, default=512)

    # -- sbert --
    p_sbert = subparsers.add_parser("sbert", help="SentenceBERT text embeddings")
    _add_common_args(p_sbert)
    p_sbert.add_argument("--postfix", type=str, default="filtered_20")
    p_sbert.add_argument("--model_name", type=str, default="bert-base-nli-stsb-mean-tokens")
    p_sbert.add_argument("--batch_size", type=int, default=256)

    # -- clip-ach-text --
    p_clip_ach = subparsers.add_parser(
        "clip-ach-text", help="CrafterCLIP text embeddings (attn_achievement)"
    )
    _add_common_args(p_clip_ach)
    p_clip_ach.add_argument("--postfix", type=str, default="filtered_20")
    p_clip_ach.add_argument("--ckpt", type=str, default="models/mineclip/attn_achievement.weights")
    p_clip_ach.add_argument("--batch_size", type=int, default=4096)

    # -- paired --
    p_paired = subparsers.add_parser("paired", help="Paired text+video embeddings for CVAE prior")
    _add_common_args(p_paired)
    p_paired.add_argument("--ckpt", type=str, default="models/mineclip/attn_achievement.weights")
    p_paired.add_argument("--vid_batch_size", type=int, default=64)
    p_paired.add_argument("--text_batch_size", type=int, default=4096)
    p_paired.add_argument("--output", type=str, default="models/steve1/text_embeddings_ach.npz")

    # -- text --
    p_text = subparsers.add_parser("text", help="Text+video embeddings from caption_v2")
    _add_common_args(p_text)
    p_text.add_argument("--batch_size", type=int, default=512)
    p_text.add_argument("--ckpt", type=str, default=None)
    p_text.add_argument("--postfix", type=str, default="")

    args = parser.parse_args(argv)

    dispatch = {
        "clip": extract_clip,
        "sbert": extract_sbert,
        "clip-ach-text": extract_clip_ach_text,
        "paired": extract_paired,
        "text": extract_text,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main_cli()
