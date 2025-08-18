import os
import logging
import numpy as np
import json
import yaml
import random
from datetime import datetime
from tqdm import tqdm
from omegaconf import DictConfig

import torch
import torch.nn as nn
import torch.optim as optim

from lightning.fabric import Fabric
from torch.utils.data import DataLoader

from crafterdojo.common.train_helper import print_model_parameters
from crafterdojo.common.train_helper import save_checkpoint, compute_gradient_l2_norm, compute_weight_l2_norm
from crafterdojo.data.clipcaption_dataset import ClipCaptionDataset
from crafterdojo.model.crafterclip.mineclip import MineCLIP
from crafterdojo.model.crafterclip.utils.info_nce import info_nce
from crafterdojo.model.crafterclip.utils.metrics import compute_metrics
from crafterdojo.common.helpers import Timer


def get_param_groups(model: MineCLIP, config: DictConfig) -> list[dict]:
    base_lr = config.learning_rate
    layerwise_lr_decay = config.layerwise_lr_decay
    pretrained_layers_lr_multiplier = config.pretrained_layers_lr_multiplier

    vision_model_name = "image_encoder.blocks"
    text_model_name = "clip_model.text_model.blocks"

    param_groups: dict[float, list[torch.nn.Parameter]] = {}

    def _add_param(lr, param, param_groups):
        if lr not in param_groups:
            param_groups[lr] = []
        param_groups[lr].append(param)

    def _calculate_lr(
        name, blocks, base_lr, layerwise_lr_decay, pretrained_layers_lr_multiplier
    ):
        try:
            tokens = name.split(".")
            block_idx = int(tokens[tokens.index("blocks") + 1])
        except Exception as e:
            block_idx = None

        total_blocks = len(blocks)
        if (block_idx is not None) and (block_idx < total_blocks):
            exponent = 0 if block_idx == (total_blocks - 1) else 1
            return (
                base_lr
                * pretrained_layers_lr_multiplier
                * (layerwise_lr_decay**exponent)
            )
        return base_lr

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if vision_model_name in name:
            block_lr = _calculate_lr(
                name,
                model.image_encoder.blocks,
                base_lr,
                layerwise_lr_decay,
                pretrained_layers_lr_multiplier,
            )
        elif text_model_name in name:
            block_lr = _calculate_lr(
                name,
                model.clip_model.text_model.blocks,
                base_lr,
                layerwise_lr_decay,
                pretrained_layers_lr_multiplier,
            )
        else:
            block_lr = base_lr

        _add_param(block_lr, param, param_groups)

    final_param_groups = [
        {"params": params, "lr": lr} for lr, params in param_groups.items()
    ]
    return final_param_groups


def process_batch(model: MineCLIP, fabric: Fabric, videos, texts):
    videos, texts = fabric.to_device(videos), fabric.to_device(texts)

    video_feats_batch = model.encode_video(videos)
    text_feats_batch = model.encode_text(texts)
    assert video_feats_batch.shape == text_feats_batch.shape

    return video_feats_batch, text_feats_batch


@torch.no_grad()
def run_validation(config, model, val_dataloader, device, fabric: Fabric):
    sum_loss = 0
    n_loss = 0

    metrics = {"R1": 0, "R5": 0, "R10": 0, "MR": 0, "MeanR": 0}
    n_metrics = 0

    val_batch = 0
    total_batch = len(val_dataloader)

    model.eval()
    for batch in (
        tqdm(val_dataloader, total=total_batch, desc="Validation")
        if fabric.is_global_zero
        else val_dataloader
    ):
        val_batch += 1

        videos, texts, _ = batch
        video_feats_batch, text_feats_batch = process_batch(
            model, fabric, videos, texts
        )

        unique_texts = sorted(set(texts))
        text_to_idx = {text: idx for idx, text in enumerate(unique_texts)}
        label_indices = torch.tensor([text_to_idx[text] for text in texts],
                                     device=fabric.device)

        logits_per_video, _ = model(
            video_feats_batch,
            text_tokens=text_feats_batch,
            is_video_features=True,
        )

        loss = info_nce(logits_per_video, label_indices)

        loss = fabric.all_reduce(loss, reduce_op="mean").item()
        sum_loss += loss
        n_loss += 1

        vt_metrics = compute_metrics(logits_per_video, label_indices)

        R1 = fabric.all_reduce(vt_metrics["R1"], reduce_op="sum")
        R5 = fabric.all_reduce(vt_metrics["R5"], reduce_op="sum")
        R10 = fabric.all_reduce(vt_metrics["R10"], reduce_op="sum")
        MR = fabric.all_reduce(vt_metrics["MR"], reduce_op="sum")
        MeanR = fabric.all_reduce(vt_metrics["MeanR"], reduce_op="sum")

        metrics["R1"] += R1
        metrics["R5"] += R5
        metrics["R10"] += R10
        metrics["MR"] += MR
        metrics["MeanR"] += MeanR
        n_metrics += fabric.world_size

        fabric.barrier()

    avg_loss = sum_loss / n_loss
    metrics = {f"val_{k}": v / n_metrics for k, v in metrics.items()}
    return avg_loss, metrics


def train_entry(config, logdir: str, fabric: Fabric):
    checkpoint_dir = os.path.join(logdir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    device = fabric.device

    ############################################################
    # Prepare dataset
    ############################################################
    video_resolution = config.resolution

    train_dataset = ClipCaptionDataset(
        config.train_dataset.dataset_root,
        config.train_dataset.caption_path,
        "train",
        config.train_dataset.data_split,
        video_resolution,
        episode_cutoff=config.train_dataset.episode_cutoff,
        augmentation=True,
    )

    val_dataset = ClipCaptionDataset(
        config.val_dataset.dataset_root,
        config.val_dataset.caption_path,
        "val",
        config.val_dataset.data_split,
        video_resolution,
        episode_cutoff=config.val_dataset.episode_cutoff,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        collate_fn=train_dataset.collate_fn,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
        collate_fn=val_dataset.collate_fn,
    )

    if config.rephrase_dict is not None:
        with open(config.rephrase_dict, "rt") as f:
            rephrase_dict = yaml.load(f, Loader=yaml.FullLoader)

        if config.num_rephrased >= 0:
            rephrase_dict = {
                k: v[:config.num_rephrased]
                for k, v in rephrase_dict.items()
            }
    else:
        rephrase_dict = {}

    ############################################################
    # Prepare model
    ############################################################
    model = MineCLIP(**config.mineclip)
    if config.oai_clip_weights is not None:
        clip_state_dict = torch.load(config.oai_clip_weights)
        model.clip_model.load_state_dict(clip_state_dict)

    def set_requires_grad(module, requires_grad: bool = True):
        for param in module.parameters():
            param.requires_grad = requires_grad

    set_requires_grad(model.clip_model.vision_model, False)
    for block in model.clip_model.vision_model.blocks[-2:]:
        set_requires_grad(block, True)

    set_requires_grad(model.clip_model.text_model, False)
    for block in model.clip_model.text_model.blocks[-2:]:
        set_requires_grad(block, True)

    print_model_parameters(model, fabric)

    param_groups = get_param_groups(model, config)
    optimizer = optim.AdamW(
        params=param_groups,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer=optimizer, total_iters=config.warmup_steps
    )    
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=(len(train_dataloader) * config.n_epochs)
        - config.warmup_steps,
        eta_min=config.min_learning_rate,
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer=optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[config.warmup_steps],
    )

    ############################################################
    # Prepare trainer
    ############################################################
    model, optimizer = fabric.setup(model, optimizer)
    model.mark_forward_method("encode_video")
    model.mark_forward_method("encode_text")

    train_dataloader, val_dataloader = fabric.setup_dataloaders(
        train_dataloader, val_dataloader
    )

    ############################################################
    # Training loop
    ############################################################
    best_val_loss = float("inf")

    n_batches, n_steps, sum_loss, n_loss, sum_gt = 0, 0, 0, 0, 0
    metrics = {"R1": 0, "R5": 0, "R10": 0, "MR": 0, "MeanR": 0}

    timer = Timer("timings")

    real_batch_size = config.batch_size * fabric.world_size
    epoch_len = len(train_dataloader)

    val_freq = config.val_freq

    if fabric.is_global_zero:
        logging.info(f"Training started at {datetime.now()}...")

    n_batches_per_epoch = len(train_dataloader)

    if fabric.is_global_zero:
        logging.info(f"Number of batches per epoch: {n_batches_per_epoch:,}")

    next_snapshot_n_frames = config.snapshot_every_n_frames

    def save_checkpoint_weights(path: str):
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "n_steps": n_steps,
        }
        fabric.save(path, checkpoint)

    def run_validation_internal():
        nonlocal best_val_loss
        if fabric.is_global_zero:
            logging.info(f"Validation at step {n_steps}...")
        with timer.time("validation"):
            val_loss, metrics = run_validation(config, model, val_dataloader, device, fabric)
        metrics_log.update(
            {
                "val_loss": val_loss,
                **metrics
            }
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            if fabric.is_global_zero:
                logging.info(f"New best validation loss: {best_val_loss:.4f}")
            save_checkpoint_weights(os.path.join(checkpoint_dir, "best.weights"))

            if fabric.is_global_zero:
                metadata = {
                    "best_val_loss": best_val_loss,
                    "n_steps": n_steps,
                    "epoch": n_steps / epoch_len,
                }
                with open(
                    os.path.join(checkpoint_dir, "best_metadata.json"), "wt"
                ) as f:
                    json.dump(metadata, f, indent=4)


    for _ in range(config.n_epochs):
        model.train()

        for batch in train_dataloader:
            n_batches += 1

            timer.throughput("samples", real_batch_size)

            metrics_log = {}

            with timer.time("data prep"):
                videos, texts, _ = batch

                rephrase_text = [
                    random.choice(rephrase_dict.get(text, []) + [text])
                    for text in texts
                ]
                video_feats_batch, text_feats_batch = process_batch(
                    model, fabric, videos, rephrase_text
                )

                unique_texts = sorted(set(texts))
                text_to_idx = {text: idx for idx, text in enumerate(unique_texts)}
                label_indices = torch.tensor([text_to_idx[text] for text in texts],
                                             device=fabric.device)

            with timer.time("train forward"):
                logits_per_video, _ = model(
                    video_feats_batch,
                    text_tokens=text_feats_batch,
                    is_video_features=True
                )
                loss = info_nce(logits_per_video, label_indices)
                sum_gt += fabric.all_reduce(torch.diag(logits_per_video).mean(), reduce_op="mean").item()

                vt_metrics = compute_metrics(logits_per_video, label_indices)

                metrics["R1"] += fabric.all_reduce(vt_metrics["R1"], reduce_op="mean")
                metrics["R5"] += fabric.all_reduce(vt_metrics["R5"], reduce_op="mean")
                metrics["R10"] += fabric.all_reduce(vt_metrics["R10"], reduce_op="mean")
                metrics["MR"] += fabric.all_reduce(vt_metrics["MR"], reduce_op="mean")
                metrics["MeanR"] += fabric.all_reduce(vt_metrics["MeanR"], reduce_op="mean")

            with timer.time("train backward"):
                optimizer.zero_grad()
                fabric.backward(loss)

            with timer.time("train optimizer step"):
                optimizer.step()
                scheduler.step()

            n_steps += 1
            sum_loss += loss.item()
            n_loss += 1

            # Log training metrics
            if (n_steps - 1) % config.log_freq == 0 or (
                    n_steps - 1
                ) % val_freq == 0:
                    avg_loss = sum_loss / n_loss

                    metrics_log.update(
                        {
                            "loss": avg_loss,
                            "step": n_steps,
                            "processed_samples": n_steps * real_batch_size,
                            "epoch": n_steps / epoch_len,
                            "learning_rate": optimizer.param_groups[0]["lr"],
                            "grad_l2_norm": compute_gradient_l2_norm(model),
                            "weight_l2_norm": compute_weight_l2_norm(model),
                            "gt_score": sum_gt / n_loss,

                            **{k: v / n_loss for k, v in metrics.items()}
                        }
                    )
                    sum_loss, n_loss = 0, 0
                    sum_gt = 0
                    metrics = {"R1": 0, "R5": 0, "R10": 0, "MR": 0, "MeanR": 0}

            # Save model checkpoint
            if (n_steps - 1) % config.save_freq == 0:
                save_checkpoint_weights(os.path.join(checkpoint_dir, "last.weights"))
                save_checkpoint(fabric, checkpoint_dir, n_batches, best_val_loss)

            if (n_steps * real_batch_size) >= next_snapshot_n_frames:
                save_checkpoint_weights(os.path.join(checkpoint_dir, f"snapshot_{n_steps}.weights"))
                next_snapshot_n_frames += config.snapshot_every_n_frames

            # Validation step
            if (n_steps - 1) % val_freq == 0:
                run_validation_internal()

            if metrics_log:
                metrics_log.update(timer.dict())
                timer.reset()
                if fabric.is_global_zero:
                    fabric.log_dict(metrics_log, step=n_steps)
                    logging.info(f"Metrics for step {n_steps}:")
                    logging.info(f"\tCurrent Datetime: {datetime.now()}")
                    for k, v in metrics_log.items():
                        logging.info(f"\t{k}: {v}")

        run_validation_internal()
        metrics_log.update(timer.dict())
        timer.reset()
        if fabric.is_global_zero:
            fabric.log_dict(metrics_log, step=n_steps)
            logging.info(f"Metrics for step {n_steps}:")
            logging.info(f"\tCurrent Datetime: {datetime.now()}")
            for k, v in metrics_log.items():
                logging.info(f"\t{k}: {v}")


@torch.no_grad()
def eval_entry(config, logdir: str, fabric: Fabric):
    ############################################################
    # Prepare dataset
    ############################################################
    video_resolution = config.resolution

    test_dataset_root = config.test_dataset.dataset_root
    test_dataset_split = config.test_dataset.data_split
    test_dataset_episode_cutoff = config.test_dataset.episode_cutoff

    test_dataset = ClipCaptionDataset(
        test_dataset_root,
        config.test_dataset.caption_path,
        "val",
        test_dataset_split,
        video_resolution,
        episode_cutoff=test_dataset_episode_cutoff,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
        collate_fn=test_dataset.collate_fn,
    )

    ############################################################
    # Prepare model
    ############################################################
    model = MineCLIP(**config.mineclip)

    assert config.weights is not None
    clip_state_dict = torch.load(config.weights)["state_dict"]
    model.load_state_dict(clip_state_dict)

    model = fabric.setup(model)
    model.mark_forward_method("encode_video")
    model.mark_forward_method("encode_text")

    test_dataloader = fabric.setup_dataloaders(test_dataloader)

    ############################################################
    # Evaluation loop
    ############################################################
    metrics = {"R1": 0, "R5": 0, "R10": 0, "MR": 0, "MeanR": 0}
    gt_scores = 0
    counts = 0

    model.eval()

    total_batch = len(test_dataloader)

    print(f"rank: {fabric.global_rank}, batch size: {config.batch_size}")

    for batch in (
        tqdm(test_dataloader, total=total_batch, desc="Validation")
        if fabric.is_global_zero
        else test_dataloader
    ):
        videos, texts, _ = batch
        video_feats_batch, text_feats_batch = process_batch(
            model, fabric, videos, texts
        )

        unique_texts = sorted(set(texts))
        text_to_idx = {text: idx for idx, text in enumerate(unique_texts)}
        label_indices = torch.tensor([text_to_idx[text] for text in texts],
                                     device=fabric.device)

        rewards, _ = model(
            video_feats_batch, text_tokens=text_feats_batch, is_video_features=True
        )

        # if fabric.is_global_zero:
        #     print(torch.diag(rewards))
        #     print(torch.max(rewards, dim=1)[0])
        #     print(torch.min(rewards, dim=1)[0])
        #     print()

        gt_scores += fabric.all_reduce(torch.diag(rewards).sum(), reduce_op="sum").item()
        vt_metrics = compute_metrics(rewards, label_indices)

        # if fabric.is_global_zero:
        #     print(vt_metrics["MeanR"])

        R1 = fabric.all_reduce(vt_metrics["R1"], reduce_op="sum")
        R5 = fabric.all_reduce(vt_metrics["R5"], reduce_op="sum")
        R10 = fabric.all_reduce(vt_metrics["R10"], reduce_op="sum")
        MR = fabric.all_reduce(vt_metrics["MR"], reduce_op="sum")
        MeanR = fabric.all_reduce(vt_metrics["MeanR"], reduce_op="sum")

        metrics["R1"] += R1
        metrics["R5"] += R5
        metrics["R10"] += R10
        metrics["MR"] += MR
        metrics["MeanR"] += MeanR
        counts += fabric.world_size

    if fabric.is_global_zero:
        metrics = {k: v / counts for k, v in metrics.items()}
        logging.info(
            f"R@1: {metrics['R1']:.4f} - R@5: {metrics['R5']:.4f} - R@10: {metrics['R10']:.4f} - Median R: {metrics['MR']:.4f} - Mean R: {metrics['MeanR']:.4f}"
        )
        logging.info(f"GT Score: {gt_scores / counts / config.batch_size / fabric.world_size:.4f}")
