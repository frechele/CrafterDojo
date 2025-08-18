import os
import math
from tqdm import tqdm
import inspect
import pickle
import json
import numpy as np
import logging
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.distributions import Categorical
from lightning.fabric import Fabric

from crafterdojo.common.train_helper import configure_optimizers, get_lr, print_model_parameters
from crafterdojo.data.steve1_dataset import Steve1Dataset
from crafterdojo.lib.steve1.MineRLConditionalAgent import CrafterConditionalAgent
from crafterdojo.common.helpers import object_to_torch_and_device, Timer
from crafterdojo.lib.VPT.lib.tree_util import tree_map
from crafterdojo.lib.VPT.agent import load_model_parameters
from crafterdojo.common.train_helper import save_checkpoint, compute_gradient_l2_norm, compute_weight_l2_norm


def get_chunk(x, t, trunc_t):
    if isinstance(x, torch.Tensor):
        return x[:, t : t + trunc_t]
    else:
        return [y[t : t + trunc_t] for y in x]


def compute_entropy(pi_distribution):
    return Categorical(logits=pi_distribution).entropy().mean()


@torch.no_grad()
def run_validation(config, policy, val_dataloader, device, fabric: Fabric):
    agent_state = None
    sum_loss = 0
    sum_kl = 0
    n_loss = 0

    val_batch = 0
    total_batch = len(val_dataloader)

    trunc_t = config.trunc_t
    policy.eval()
    for obs, actions, firsts in (
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

            obs_chunk = object_to_torch_and_device(obs_chunk, device)
            actions_chunk = object_to_torch_and_device(actions_chunk, device)
            firsts_chunk = object_to_torch_and_device(firsts_chunk, device).view(
                B, trunc_t
            )

            if agent_state is None:
                agent_state = policy.initial_state(B)

            pi_distribution, _, new_agent_state = policy.get_output_for_observation(
                obs_chunk, agent_state, firsts_chunk
            )
            pi_distribution = tree_map(
                lambda x: x.view(B, trunc_t, -1), pi_distribution
            )

            log_prob = policy.get_logprob_of_action(pi_distribution, actions_chunk)
            loss = -log_prob.mean()

            uncond_obs_chunk = obs_chunk.copy()
            uncond_obs_chunk["goal_embed"] = torch.zeros_like(uncond_obs_chunk["goal_embed"])
            uncond_pi_distribution, _, _ = policy.get_output_for_observation(
                uncond_obs_chunk, agent_state, firsts_chunk
            )
            uncond_pi_distribution = tree_map(
                lambda x: x.view(B, trunc_t, -1), uncond_pi_distribution
            )

            kl_between_cond_and_uncond = policy.get_kl_of_action_dists(
                uncond_pi_distribution, pi_distribution
            )
            kl_between_cond_and_uncond = kl_between_cond_and_uncond.mean()

            agent_state = new_agent_state
            loss = fabric.all_reduce(loss, reduce_op="mean")
            sum_loss += loss.item()
            sum_kl += kl_between_cond_and_uncond.item()
            n_loss += 1

    avg_loss = sum_loss / n_loss
    avg_kl = sum_kl / n_loss
    return avg_loss, avg_kl


def train_entry(config, env_entry, logdir: str, fabric: Fabric):
    checkpoint_dir = os.path.join(logdir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    device = fabric.device

    ############################################################
    # Prepare dataset
    ############################################################
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
        episode_cutoff=train_dataset_episode_cutoff,
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
        episode_cutoff=val_dataset_episode_cutoff,
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
    agent_policy_kwargs, agent_pi_head_kwargs, lora_kwargs = load_model_parameters(config.model)
    agent = CrafterConditionalAgent(device, agent_policy_kwargs, agent_pi_head_kwargs, lora_kwargs)

    if config.weights is not None:
        agent.load_weights(config.weights)
    else:
        agent.policy.reset_parameters()

    policy = agent.policy

    # for name, param in policy.net.named_parameters():
    #     if "mineclip_embed_linear" in name:
    #     # if "img_process" not in name:
    #         param.requires_grad = True
    #     else:
    #         param.requires_grad = False
    # disable value head gradient, as we don't need it for BC
    for _, param in policy.value_head.named_parameters():
        param.requires_grad = False

    print_model_parameters(policy, fabric)

    optimizer = configure_optimizers(policy, config.weight_decay, config.learning_rate)

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

    timer = Timer("timings")

    real_batch_size = config.batch_size * fabric.world_size
    epoch_len = len(train_dataloader) * (config.T // config.trunc_t)

    frames_per_step = config.trunc_t * real_batch_size
    if fabric.is_global_zero:
        logging.info(f"Batch size in frames: {frames_per_step:,}")

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
        config.n_samples, config.n_epochs * epoch_len * frames_per_step
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

    while n_steps * frames_per_step <= frame_limit and not end_training:
        policy.train()
        for obs, actions, firsts in timer.time_iter(train_dataloader, "dataloader"):
            n_batches += 1
            if n_steps * frames_per_step >= frame_limit:
                end_training = True
                break

            B = firsts.shape[0]
            T = firsts.shape[1]
            timer.throughput("frames", T * real_batch_size)

            for t in range(0, T, trunc_t):
                num_frames_processed = n_steps * frames_per_step
                lr = get_lr(config, num_frames_processed)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

                metrics_log = {}

                with timer.time("data prep"):
                    obs_chunk = tree_map(lambda x: get_chunk(x, t, trunc_t), obs)
                    actions_chunk = tree_map(
                        lambda x: get_chunk(x, t, trunc_t), actions
                    )
                    firsts_chunk = firsts[:, t : t + trunc_t]

                    obs_chunk = object_to_torch_and_device(obs_chunk, device)
                    actions_chunk = object_to_torch_and_device(actions_chunk, device)
                    firsts_chunk = object_to_torch_and_device(
                        firsts_chunk, device
                    ).view(B, trunc_t)

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
                    loss = -log_prob.mean()

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
                            "processed_frames": n_steps * frames_per_step,
                            "epoch": n_steps / epoch_len,
                            "learning_rate": lr,
                            "grad_l2_norm": compute_gradient_l2_norm(policy),
                            "weight_l2_norm": compute_weight_l2_norm(policy),
                            "goal_grad_l2_norm": compute_gradient_l2_norm(policy.net.mineclip_embed_linear),
                            "goal_weight_l2_norm": compute_weight_l2_norm(policy.net.mineclip_embed_linear),
                            "accuracy": (
                                pi_distribution.argmax(dim=-1) == actions_chunk
                            )
                            .float()
                            .mean(),
                            "entropy": compute_entropy(pi_distribution),
                        }
                    )
                    sum_loss, n_loss = 0, 0

                # Save model checkpoint
                if (n_steps - 1) % config.save_freq == 0:
                    state_dict = policy.state_dict()
                    fabric.save(
                        os.path.join(checkpoint_dir, "last.weights"), state_dict
                    )
                    save_checkpoint(fabric, checkpoint_dir, n_batches, best_val_loss)

                if (n_steps * frames_per_step) >= next_snapshot_n_samples:
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

    run_validation_internal()
    metrics_log.update(timer.dict())
    timer.reset()
    if fabric.is_global_zero:
        fabric.log_dict(metrics_log, step=n_steps)
        logging.info(f"Metrics for step {n_steps}:")
        logging.info(f"\tCurrent Datetime: {datetime.now()}")
        for k, v in metrics_log.items():
            logging.info(f"\t{k}: {v}")
