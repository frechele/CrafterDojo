import argparse
import os
import numpy as np
import pickle
from tqdm import tqdm

import torch

from crafterdojo.data.steve1_dataset import Episode
from crafterdojo.model.crafterclip import load_mineclip_wconfig, CRAFTER_MINECLIP_CONFIG


NONE_EMBED_OFFSET = 5


@torch.no_grad()
def main(args):
    device = torch.device("cuda:0")
    clip_config = CRAFTER_MINECLIP_CONFIG.copy()
    if args.ckpt is not None:
        clip_config["ckpt"] = args.ckpt
    mineclip = load_mineclip_wconfig(device, cfg=clip_config)
    mineclip.eval()

    B = args.batch_size

    results = {
        "captions": None,
        "text_embeddings": None,
        "vid_embeddings": None,
    }

    for episode_id in tqdm(range(args.start_ep, args.start_ep + args.n_episodes)):
        episode = Episode(args.root, episode_id)
        episode_len = len(episode)

        caption_path = os.path.join(args.root, "caption", f"episode_{episode_id:05d}.pkl")
        with open(caption_path, "rb") as f:
            captions = pickle.load(f)

        frames = []
        caption_texts = []
        for caption in captions:
            start_timestep, end_timestep = caption[0]
            caption_text = caption[1]

            if end_timestep <= NONE_EMBED_OFFSET:
                continue

            frames.append(episode.load_frames(only_range=(start_timestep, end_timestep))[start_timestep:end_timestep])

            caption_texts.append(caption_text)

        for batch_start in range(0, len(caption_texts), B):
            batch_end = min(batch_start + B, len(caption_texts))

            batch_texts = caption_texts[batch_start:batch_end]

            local_frames = np.stack(frames[batch_start:batch_end]).transpose(0, 1, 4, 2, 3)
            embeddings = mineclip.encode_video(torch.from_numpy(local_frames).to(device)).cpu().numpy()
            batch_embeddings = mineclip.encode_text(batch_texts).cpu().numpy()

            if results["text_embeddings"] is None:
                results["text_embeddings"] = batch_embeddings
                results["vid_embeddings"] = embeddings
                results["captions"] = np.array(batch_texts)
            else:
                results["text_embeddings"] = np.concatenate(
                    [results["text_embeddings"], batch_embeddings], axis=0)
                results["vid_embeddings"] = np.concatenate(
                    [results["vid_embeddings"], embeddings], axis=0)
                results["captions"] = np.concatenate(
                    [results["captions"], np.array(batch_texts)], axis=0)

    print({k: v.shape for k, v in results.items()})

    postfix = f"_{args.postfix}" if args.postfix else ""
    out_filename = f"text_embeddings{postfix}.npz"
    np.savez(os.path.join(args.root, out_filename), **results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=str)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--start_ep", type=int, default=0)
    parser.add_argument("--n_episodes", type=int, default=1000)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--postfix", type=str, default="")

    args = parser.parse_args()
    main(args)
