import argparse
import cv2
import os
import numpy as np
from tqdm import tqdm

import torch

from crafterdojo.data.vpt_dataset import Episode
from crafterdojo.model.crafterclip import load_mineclip_wconfig


WINDOW_SIZE = 6


def image_preprocess(frame):
    return (cv2.resize(frame, (144, 144))
               .transpose(2, 0, 1))


@torch.no_grad()
def main(args):
    embedding_root = os.path.join(args.root, "clip_embedding")
    os.makedirs(embedding_root, exist_ok=True)

    device = torch.device("cuda:0")
    mineclip = load_mineclip_wconfig(device)
    mineclip.eval()

    B = args.batch_size

    for episode_id in tqdm(range(args.start_ep, args.start_ep + args.n_episodes)):
        episode = Episode(args.root, episode_id)

        episode_embeddings = np.zeros((WINDOW_SIZE - 1, 512), dtype=np.float32)
        n_frames = len(episode) + 1
        for start_idx in tqdm(range(0, n_frames - WINDOW_SIZE + 1, B)):
            real_batch = min(B, n_frames - start_idx - WINDOW_SIZE + 1)

            end_idx = start_idx + (real_batch - 1) + WINDOW_SIZE
            frames = episode.load_frames((start_idx, end_idx))[start_idx:end_idx]

            all_frames = np.array([image_preprocess(frame) for frame in frames])
            indices = np.arange(real_batch)[:, None] + np.arange(WINDOW_SIZE)[None, :]
            batch = all_frames[indices].astype(np.float32)
            batch = torch.from_numpy(batch).to(device)

            embeddings = mineclip.encode_video(batch).cpu().numpy()
            episode_embeddings = np.concatenate([episode_embeddings, embeddings], axis=0)

        print("episode length", len(episode))
        print("episode_embeddings.shape", episode_embeddings.shape)

        np.savez_compressed(
            os.path.join(embedding_root, f"{episode_id:05d}.npz"),
            window_size=WINDOW_SIZE,
            embeddings=episode_embeddings,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=str)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--start_ep", type=int, default=0)
    parser.add_argument("--n_episodes", type=int, default=20000)

    args = parser.parse_args()
    main(args)
