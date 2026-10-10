import json
import os

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from crafterdojo.data.clipcaption_dataset import ClipCaptionDataset
from crafterdojo.evaluator.crafterclip.openai import OpenAICLIPEvaluator
from crafterdojo.evaluator.crafterclip.retrieval import (
    compute_retrieval_metrics,
    normalize_texts_join,
)


@torch.no_grad()
def evaluate_openai(
    dataset_root,
    caption_path,
    data_split,
    batch_size=32,
    num_workers=4,
    episode_cutoff=None,
    device="cuda",
    output_file=None,
    weights_path=None,
    image_resolution=(144, 144),
):
    if isinstance(data_split, str) and not data_split.strip().startswith("["):
        data_split = f"[{data_split}]"

    evaluator = OpenAICLIPEvaluator(
        device=device,
        weights_path=weights_path,
        image_resolution=image_resolution,
    )
    video_resolution = evaluator.image_resolution

    dataset = ClipCaptionDataset(
        dataset_root=dataset_root,
        caption_path=caption_path,
        mode="val",
        data_split=data_split,
        resize_resolution=video_resolution,
        episode_cutoff=episode_cutoff,
        augmentation=False,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        collate_fn=dataset.collate_fn,
    )

    with open("data/skillbook.yaml", "rt") as f:
        all_texts = yaml.load(f, Loader=yaml.FullLoader)
    if not isinstance(all_texts, list):
        raise ValueError("skillbook.yaml must contain a list of texts.")
    normalized_texts = normalize_texts_join(all_texts)
    text_to_idx = {text: idx for idx, text in enumerate(normalized_texts)}
    text_features = evaluator.extract_text_features(normalized_texts)

    print(f"Evaluating on {len(dataset)} samples...")

    total_metrics = {
        "v2t_r1": 0, "v2t_r5": 0, "v2t_r10": 0, "v2t_medr": 0, "v2t_meanr": 0,
        "t2v_r1": 0, "t2v_r5": 0, "t2v_r10": 0, "t2v_medr": 0, "t2v_meanr": 0,
    }
    n_batches = 0

    for videos, texts, _ in tqdm(dataloader, desc="Computing metrics"):
        videos = videos.to(device)
        video_features = evaluator.extract_video_features(videos)

        normalized_batch = normalize_texts_join(texts)
        missing_texts = [text for text in normalized_batch if text not in text_to_idx]
        if missing_texts:
            missing_texts = sorted(set(missing_texts))
            raise ValueError(f"Missing texts in skillbook.yaml: {missing_texts}")
        label_indices = torch.tensor(
            [text_to_idx[text] for text in normalized_batch], device=device
        )

        similarity_matrix = evaluator.compute_similarity_matrix(video_features, text_features)
        batch_metrics = compute_retrieval_metrics(similarity_matrix, label_indices)

        for key in total_metrics:
            total_metrics[key] += batch_metrics[key]
        n_batches += 1

    for key in total_metrics:
        total_metrics[key] /= n_batches

    print("\n" + "=" * 50)
    print("OpenAI CLIP Evaluation Results")
    print("=" * 50)
    print("Video-to-Text Retrieval:")
    print(f"  R@1:  {total_metrics['v2t_r1']:.4f}")
    print(f"  R@5:  {total_metrics['v2t_r5']:.4f}")
    print(f"  R@10: {total_metrics['v2t_r10']:.4f}")
    print(f"  MedR: {total_metrics['v2t_medr']:.2f}")
    print(f"  MeanR: {total_metrics['v2t_meanr']:.2f}")
    print("\nText-to-Video Retrieval:")
    print(f"  R@1:  {total_metrics['t2v_r1']:.4f}")
    print(f"  R@5:  {total_metrics['t2v_r5']:.4f}")
    print(f"  R@10: {total_metrics['t2v_r10']:.4f}")
    print(f"  MedR: {total_metrics['t2v_medr']:.2f}")
    print(f"  MeanR: {total_metrics['t2v_meanr']:.2f}")

    if output_file:
        results = {
            "model": "openai/clip-vit-base-patch16",
            "dataset_info": {
                "dataset_root": dataset_root,
                "caption_path": caption_path,
                "data_split": data_split,
                "num_samples": len(dataset),
            },
            "weights_path": evaluator.weights_path,
            "metrics": total_metrics,
        }
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")

    return total_metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OpenAI CLIP Evaluation on CrafterDojo Dataset")
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--caption_path", type=str, default="captions")
    parser.add_argument("--data_split", type=str, default="0:1000")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--episode_cutoff", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--weights_path", type=str, default=None)
    args = parser.parse_args()

    evaluate_openai(
        dataset_root=args.dataset_root,
        caption_path=args.caption_path,
        data_split=args.data_split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        episode_cutoff=args.episode_cutoff,
        device=args.device,
        output_file=args.output_file,
        weights_path=args.weights_path,
    )
