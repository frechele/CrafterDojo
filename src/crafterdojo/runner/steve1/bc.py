import os
from tqdm import tqdm
import json
import numpy as np
import logging
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from lightning.fabric import Fabric

from crafterdojo.agent.steve1.cfg_metrics import compute_cfg_kl_sum
from crafterdojo.agent.steve1.factory import build_conditional_agent
from crafterdojo.common.train_helper import configure_optimizers, get_lr, print_model_parameters
from crafterdojo.data.steve1_dataset import Steve1Dataset
from crafterdojo.common.helpers import object_to_torch_and_device, Timer
from crafterdojo.common.loss_utils import (
    categorical_accuracy_sum,
    categorical_entropy_sum,
    get_chunk,
    masked_sum,
)
from crafterdojo.model.vpt.lib.tree_util import tree_map
from crafterdojo.common.train_helper import save_checkpoint, compute_gradient_l2_norm, compute_weight_l2_norm


@torch.no_grad()
def run_validation(config, policy, val_dataloader, device, fabric: Fabric):
    agent_state = None
    sum_loss = 0
    sum_kl = 0
    n_valid = 0

    val_batch = 0
    total_batch = len(val_dataloader)

    trunc_t = config.trunc_t
    policy.eval()
    for obs, actions, firsts, valid_masks in (
        tqdm(val_dataloader, total=total_batch, desc="Validation")
        if fabric.is_global_zero
        else val_dataloader
    ):
        val_batch += 1
        B = firsts.shape[0]
        T = firsts.shape[1]

        for t in range(0, T, trunc_t):
            obs_chunk = tree_map(lambda x: get_chunk(x, t, trunc_t), obs)
            actions_chunk = get_chunk(actions, t, trunc_t)
            firsts_chunk = firsts[:, t : t + trunc_t]
            valid_mask_chunk = valid_masks[:, t : t + trunc_t]

            obs_chunk = object_to_torch_and_device(obs_chunk, device)
            actions_chunk = object_to_torch_and_device(actions_chunk, device)
            firsts_chunk = object_to_torch_and_device(firsts_chunk, device).view(
                B, trunc_t
            )
            valid_mask_chunk = object_to_torch_and_device(
                valid_mask_chunk, device
            ).view(B, trunc_t)

            local_valid_count = valid_mask_chunk.sum().to(dtype=torch.float32)
            global_valid_count = fabric.all_reduce(
                local_valid_count, reduce_op="sum"
            )
            global_valid_count_value = int(global_valid_count.item())
            if global_valid_count_value == 0:
                break

            if agent_state is None:
                agent_state = policy.initial_state(B)

            pi_distribution, _, new_agent_state = policy.get_output_for_observation(
                obs_chunk, agent_state, firsts_chunk
            )
            pi_distribution = tree_map(
                lambda x: x.view(B, trunc_t, -1), pi_distribution
            )

            log_prob = policy.get_logprob_of_action(pi_distribution, actions_chunk)
            loss_sum = -masked_sum(log_prob, valid_mask_chunk)

            kl_sum = compute_cfg_kl_sum(
                policy,
                obs_chunk,
                agent_state,
                firsts_chunk,
                pi_distribution,
                valid_mask_chunk,
                (B, trunc_t),
            )

            agent_state = new_agent_state
            loss_sum = fabric.all_reduce(loss_sum.detach(), reduce_op="sum")
            kl_sum = fabric.all_reduce(kl_sum.detach(), reduce_op="sum")
            sum_loss += loss_sum.item()
            sum_kl += kl_sum.item()
            n_valid += global_valid_count_value

    avg_loss = sum_loss / max(n_valid, 1)
    avg_kl = sum_kl / max(n_valid, 1)
    return avg_loss, avg_kl


def main(config, logdir: str, fabric: Fabric, env_entry=None):
    checkpoint_dir = os.path.join(logdir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    device = fabric.device

    ############################################################
    # Prepare dataset
    ############################################################
    clip_rephrased = config.rephrased
    embed_type = getattr(config, "embed_type", "clip")
    train_dataset_root = config.train_dataset.dataset_root
    train_dataset_split = config.train_dataset.data_split
    train_dataset_episode_cutoff = config.train_dataset.episode_cutoff
    train_dataset = Steve1Dataset(
        train_dataset_root,
        train_dataset_split,
        config.T,
        config.min_btwn_goals,
        config.max_btwn_goals,
        config.p_uncond,
        config.event_based_goals,
        rephrased=clip_rephrased,
        episode_cutoff=train_dataset_episode_cutoff,
        embed_type=embed_type,
        include_partial_chunks=config.include_partial_chunks,
    )

    val_dataset_root = config.val_dataset.dataset_root
    val_dataset_split = config.val_dataset.data_split
    val_dataset_episode_cutoff = config.val_dataset.episode_cutoff
    val_dataset = Steve1Dataset(
        val_dataset_root,
        val_dataset_split,
        config.T,
        config.min_btwn_goals,
        config.max_btwn_goals,
        config.p_uncond,
        config.event_based_goals,
        rephrased=clip_rephrased,
        episode_cutoff=val_dataset_episode_cutoff,
        embed_type=embed_type,
        include_partial_chunks=config.include_partial_chunks,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
    )

    ############################################################
    # Prepare model
    ############################################################
    agent = build_conditional_agent(config, device, freeze_value_head=True)
    policy = agent.policy

    print_model_parameters(policy, fabric)

    optimizer = configure_optimizers(
        policy,
        config.weight_decay,
        config.learning_rate,
        pi_head_lr_mult=getattr(config, "pi_head_lr_mult", 1.0),
        goal_affine_lr_mult=getattr(config, "goal_affine_lr_mult", 1.0),
        lora_lr_mult=getattr(config, "lora_lr_mult", 1.0),
    )

    ############################################################
    # Prepare Trainer
    ############################################################
    policy, optimizer = fabric.setup(policy, optimizer)
    policy.mark_forward_method("get_output_for_observation")
    train_dataloader, val_dataloader = fabric.setup_dataloaders(
        train_dataloader, val_dataloader
    )

    ############################################################
    # Training loop
    ############################################################
    agent_state = None
    best_val_loss = np.inf

    n_steps, n_batches, sum_loss, n_loss = 0, 0, 0, 0
    n_valid_frames_processed = 0

    timer = Timer("timings")

    real_batch_size = config.batch_size * fabric.world_size
    epoch_len = len(train_dataloader) * (config.T // config.trunc_t)

    sequence_slots_per_step = config.trunc_t * real_batch_size
    if fabric.is_global_zero:
        logging.info(f"Batch size in sequence slots: {sequence_slots_per_step:,}")

    val_freq = config.val_freq

    if fabric.is_global_zero:
        logging.info(f"Training started at {datetime.now()}...")

    n_batches_per_epoch = len(train_dataloader)

    if fabric.is_global_zero:
        logging.info(f"Number of batches per epoch: {n_batches_per_epoch:,}")

    next_snapshot_n_samples = config.snapshot_every_n_samples

    end_training = False
    orig_n_samples = config.n_samples
    config.n_samples = max(
        config.n_samples, config.n_epochs * epoch_len * sequence_slots_per_step
    )
    frame_limit = config.n_samples

    if fabric.is_global_zero:
        logging.info(
            f"Frame limit: {frame_limit:,} (orig: {orig_n_samples:,}, n_epochs: {config.n_epochs})"
        )

    trunc_t = config.trunc_t

    def run_validation_internal():
        nonlocal best_val_loss
        if fabric.is_global_zero:
            logging.info(f"Validation at step {n_steps}...")
        with timer.time("validation"):
            val_loss, val_kl = run_validation(config, policy, val_dataloader, device, fabric)
        metrics_log.update(
            {
                "val_loss": val_loss,
                "val_kl_between_cond_and_uncond": val_kl,
            }
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if fabric.is_global_zero:
                logging.info(f"New best validation loss: {best_val_loss:.4f}")
            state_dict = policy.state_dict()
            fabric.save(os.path.join(checkpoint_dir, "best.weights"), state_dict)

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

    def save_last_checkpoint():
        state_dict = policy.state_dict()
        fabric.save(os.path.join(checkpoint_dir, "last.weights"), state_dict)
        save_checkpoint(fabric, checkpoint_dir, n_batches, best_val_loss)

    while n_valid_frames_processed <= frame_limit and not end_training:
        policy.train()
        for obs, actions, firsts, valid_masks in timer.time_iter(
            train_dataloader, "dataloader"
        ):
            n_batches += 1
            if n_valid_frames_processed >= frame_limit:
                end_training = True
                break

            B = firsts.shape[0]
            T = firsts.shape[1]
            for t in range(0, T, trunc_t):
                num_frames_processed = n_valid_frames_processed
                lr = get_lr(config, num_frames_processed)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr * param_group.get("lr_mult", 1.0)

                metrics_log = {}

                with timer.time("data prep"):
                    obs_chunk = tree_map(lambda x: get_chunk(x, t, trunc_t), obs)
                    actions_chunk = tree_map(
                        lambda x: get_chunk(x, t, trunc_t), actions
                    )
                    firsts_chunk = firsts[:, t : t + trunc_t]
                    valid_mask_chunk = valid_masks[:, t : t + trunc_t]

                    obs_chunk = object_to_torch_and_device(obs_chunk, device)
                    actions_chunk = object_to_torch_and_device(actions_chunk, device)
                    firsts_chunk = object_to_torch_and_device(
                        firsts_chunk, device
                    ).view(B, trunc_t)
                    valid_mask_chunk = object_to_torch_and_device(
                        valid_mask_chunk, device
                    ).view(B, trunc_t)

                    local_valid_count = valid_mask_chunk.sum().to(dtype=torch.float32)
                    global_valid_count = fabric.all_reduce(
                        local_valid_count, reduce_op="sum"
                    )
                    global_valid_count_value = int(global_valid_count.item())
                    if global_valid_count_value == 0:
                        break

                with timer.time("train forward"):
                    if agent_state is None:
                        agent_state = policy.initial_state(B)

                    pi_distribution, _, new_agent_state = (
                        policy.get_output_for_observation(
                            obs_chunk, agent_state, firsts_chunk
                        )
                    )
                    pi_distribution = tree_map(
                        lambda x: x.view(B, trunc_t, -1), pi_distribution
                    )

                    log_prob = policy.get_logprob_of_action(
                        pi_distribution, actions_chunk
                    )

                    surprise_stats = None
                    if getattr(config, "surprise_weight", False):
                        with torch.no_grad():
                            uncond_obs_chunk = obs_chunk.copy()
                            uncond_obs_chunk["goal_embed"] = torch.zeros_like(
                                uncond_obs_chunk["goal_embed"]
                            )
                            uncond_pi_distribution, _, _ = (
                                policy.get_output_for_observation(
                                    uncond_obs_chunk, agent_state, firsts_chunk
                                )
                            )
                            uncond_pi_distribution = tree_map(
                                lambda x: x.view(B, trunc_t, -1),
                                uncond_pi_distribution,
                            )
                            uncond_log_prob = policy.get_logprob_of_action(
                                uncond_pi_distribution, actions_chunk
                            )
                            clip_max = float(
                                getattr(config, "surprise_weight_clip", 2.83)
                            )
                            surprise = (-uncond_log_prob).clamp(0.0, clip_max)
                            local_surprise_sum = masked_sum(
                                surprise, valid_mask_chunk
                            )
                            local_valid = valid_mask_chunk.sum().clamp_min(1.0)
                            mean_surprise = (
                                local_surprise_sum / local_valid
                            ).clamp_min(1e-6)
                            weights = surprise / mean_surprise
                            local_clipped = (
                                ((-uncond_log_prob) > clip_max).to(
                                    dtype=torch.float32
                                )
                            )
                            surprise_stats = {
                                "mean": float(mean_surprise.detach().item()),
                                "max": float(weights.max().detach().item()),
                                "frac_clipped": float(
                                    masked_sum(local_clipped, valid_mask_chunk).item()
                                    / float(local_valid.item())
                                ),
                            }
                        weighted_log_prob = log_prob * weights
                        loss_sum = -masked_sum(
                            weighted_log_prob, valid_mask_chunk
                        )
                    else:
                        loss_sum = -masked_sum(log_prob, valid_mask_chunk)
                    loss = loss_sum * fabric.world_size / global_valid_count.clamp_min(
                        1.0
                    )

                with timer.time("train backward"):
                    optimizer.zero_grad()
                    fabric.backward(loss)

                with timer.time("train clip grad"):
                    fabric.clip_gradients(
                        policy, optimizer, max_norm=config.max_grad_norm
                    )

                with timer.time("train optimizer step"):
                    optimizer.step()

                agent_state = tree_map(lambda x: x.detach(), new_agent_state)

                n_steps += 1
                timer.throughput("valid_frames", global_valid_count_value)
                n_valid_frames_processed += global_valid_count_value
                reduced_loss_sum = fabric.all_reduce(loss_sum.detach(), reduce_op="sum")
                sum_loss += reduced_loss_sum.item()
                n_loss += global_valid_count_value

                # Log training metrics
                if (n_steps - 1) % config.log_freq == 0 or (
                    n_steps - 1
                ) % val_freq == 0:
                    avg_loss = sum_loss / n_loss
                    accuracy_sum = categorical_accuracy_sum(
                        pi_distribution, actions_chunk, valid_mask_chunk
                    )
                    entropy_sum = categorical_entropy_sum(
                        pi_distribution, valid_mask_chunk
                    )
                    accuracy_sum = fabric.all_reduce(accuracy_sum, reduce_op="sum")
                    entropy_sum = fabric.all_reduce(entropy_sum, reduce_op="sum")

                    metrics_log.update(
                        {
                            "loss": avg_loss,
                            "step": n_steps,
                            "processed_frames": n_valid_frames_processed,
                            "processed_sequence_slots": n_steps
                            * sequence_slots_per_step,
                            "epoch": n_steps / epoch_len,
                            "learning_rate": lr,
                            "grad_l2_norm": compute_gradient_l2_norm(policy),
                            "weight_l2_norm": compute_weight_l2_norm(policy),
                            "goal_grad_l2_norm": compute_gradient_l2_norm(policy.goal_projection),
                            "goal_weight_l2_norm": compute_weight_l2_norm(policy.goal_projection),
                            "accuracy": accuracy_sum / global_valid_count.clamp_min(1.0),
                            "entropy": entropy_sum / global_valid_count.clamp_min(1.0),
                            "valid_fraction": global_valid_count.clamp_min(0.0)
                            / sequence_slots_per_step,
                        }
                    )
                    if surprise_stats is not None:
                        metrics_log.update(
                            {
                                "surprise_mean": surprise_stats["mean"],
                                "surprise_weight_max": surprise_stats["max"],
                                "surprise_frac_clipped": surprise_stats[
                                    "frac_clipped"
                                ],
                            }
                        )
                    sum_loss, n_loss = 0, 0

                # Save model checkpoint
                if (n_steps - 1) % config.save_freq == 0:
                    save_last_checkpoint()

                if n_valid_frames_processed >= next_snapshot_n_samples:
                    state_dict = policy.state_dict()
                    fabric.save(
                        os.path.join(checkpoint_dir, f"snapshot_{n_steps}.weights"),
                        state_dict,
                    )
                    next_snapshot_n_samples += config.snapshot_every_n_samples

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

        save_last_checkpoint()
        if fabric.is_global_zero:
            logging.info(
                f"End-of-epoch checkpoint saved at step {n_steps} "
                f"(epoch {n_steps / epoch_len:.4f})"
            )

    run_validation_internal()
    save_last_checkpoint()
    metrics_log.update(timer.dict())
    timer.reset()
    if fabric.is_global_zero:
        fabric.log_dict(metrics_log, step=n_steps)
        logging.info(f"Metrics for step {n_steps}:")
        logging.info(f"\tCurrent Datetime: {datetime.now()}")
        for k, v in metrics_log.items():
            logging.info(f"\t{k}: {v}")
