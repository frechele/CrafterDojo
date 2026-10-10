import os
import time
from tqdm import tqdm
import json
import numpy as np
import logging
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from lightning.fabric import Fabric

from crafterdojo.agent.vpt.factory import build_vpt_agent
from crafterdojo.common.train_helper import configure_optimizers, get_lr, print_model_parameters
from crafterdojo.data.vpt_dataset import VPTDataset
from crafterdojo.common.helpers import object_to_torch_and_device, Timer
from crafterdojo.common.loss_utils import (
    categorical_accuracy_sum,
    categorical_entropy_sum,
    get_chunk,
    masked_sum,
)
from crafterdojo.common.train_debug import (
    find_nonfinite_gradient_parameters,
    log_nonfinite_forward_debug,
    maybe_clear_cuda_cache,
    summarize_gradient_parameters,
)
from crafterdojo.model.vpt.lib.tree_util import tree_map
from crafterdojo.common.train_helper import save_checkpoint, compute_gradient_l2_norm, compute_weight_l2_norm


def should_run_periodic(step: int, freq: int, run_first_step: bool) -> bool:
    if freq <= 0:
        return False
    if step == 1:
        return run_first_step
    return step % freq == 0


@torch.no_grad()
def run_validation(config, policy, val_dataloader, device, fabric: Fabric):
    agent_state = None
    sum_loss = 0
    n_valid = 0

    val_batch = 0
    total_batch = len(val_dataloader)

    trunc_t = config.trunc_t
    policy.eval()
    for obs, actions, firsts, _, valid_masks in (
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

            agent_state = new_agent_state
            loss_sum = fabric.all_reduce(loss_sum.detach(), reduce_op="sum")
            sum_loss += loss_sum.item()
            n_valid += global_valid_count_value

    avg_loss = sum_loss / max(n_valid, 1)
    return avg_loss


def main(config, logdir: str, fabric: Fabric, env_entry=None):
    checkpoint_dir = os.path.join(logdir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    device = fabric.device
    include_partial_chunks = bool(getattr(config, "include_partial_chunks", True))
    skip_nonfinite_updates = bool(getattr(config, "skip_nonfinite_updates", True))
    drop_noop_actions = bool(getattr(config, "drop_noop_actions", False))

    ############################################################
    # Prepare dataset
    ############################################################
    train_dataset_root = config.train_dataset.dataset_root
    train_dataset_split = config.train_dataset.data_split
    train_dataset_episode_cutoff = config.train_dataset.episode_cutoff
    train_dataset = VPTDataset(
        train_dataset_root,
        train_dataset_split,
        config.T,
        episode_cutoff=train_dataset_episode_cutoff,
        include_partial_chunks=include_partial_chunks,
        drop_noop_actions=drop_noop_actions,
    )

    val_dataset_root = config.val_dataset.dataset_root
    val_dataset_split = config.val_dataset.data_split
    val_dataset_episode_cutoff = config.val_dataset.episode_cutoff
    val_dataset = VPTDataset(
        val_dataset_root,
        val_dataset_split,
        config.T,
        episode_cutoff=val_dataset_episode_cutoff,
        every_nth=config.val_every_nth,
        include_partial_chunks=include_partial_chunks,
        drop_noop_actions=drop_noop_actions,
    )

    loader_kwargs = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": True,
        "drop_last": True,
        "persistent_workers": config.num_workers > 0 and config.persistent_workers,
    }
    if config.num_workers > 0:
        loader_kwargs["prefetch_factor"] = config.prefetch_factor

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        **loader_kwargs,
    )
    val_dataloader = DataLoader(
        val_dataset,
        shuffle=False,
        **loader_kwargs,
    )

    ############################################################
    # Prepare model
    ############################################################
    agent = build_vpt_agent(
        config, device, freeze_value_head=True, log=fabric.is_global_zero
    )
    policy = agent.policy

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
    skipped_nonfinite_updates = 0
    nonfinite_debug_logs = 0
    n_valid_samples_processed = 0
    n_sequence_slots_processed = 0

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
    epoch_step_limit = config.n_epochs * epoch_len if config.n_epochs > 0 else None
    config.n_samples = max(
        config.n_samples, config.n_epochs * epoch_len * sequence_slots_per_step
    )
    sequence_slot_limit = config.n_samples

    if fabric.is_global_zero:
        logging.info(
            f"Sequence-slot limit: {sequence_slot_limit:,} (orig: {orig_n_samples:,}, n_epochs: {config.n_epochs})"
        )
        if epoch_step_limit is not None:
            logging.info(f"Epoch step limit: {epoch_step_limit:,}")

    trunc_t = config.trunc_t

    def run_validation_internal():
        nonlocal best_val_loss
        if fabric.is_global_zero:
            logging.info(f"Validation at step {n_steps}...")
        start_time = time.time()
        val_loss = run_validation(config, policy, val_dataloader, device, fabric)
        metrics_log.update(
            {
                "val_loss": val_loss,
                "validation_time_sec": time.time() - start_time,
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

    while n_sequence_slots_processed <= sequence_slot_limit and not end_training:
        policy.train()
        for obs, actions, firsts, _, valid_masks in timer.time_iter(
            train_dataloader, "dataloader"
        ):
            n_batches += 1
            if (
                n_sequence_slots_processed >= sequence_slot_limit
                or (epoch_step_limit is not None and n_steps >= epoch_step_limit)
            ):
                end_training = True
                break

            B = firsts.shape[0]
            T = firsts.shape[1]

            for t in range(0, T, trunc_t):
                if (
                    n_sequence_slots_processed >= sequence_slot_limit
                    or (epoch_step_limit is not None and n_steps >= epoch_step_limit)
                ):
                    end_training = True
                    break

                lr = get_lr(config, n_sequence_slots_processed)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

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

                    if skip_nonfinite_updates and not torch.isfinite(
                        pi_distribution
                    ).all():
                        skipped_nonfinite_updates += 1
                        if fabric.is_global_zero:
                            logging.warning(
                                "Skipping step %s because policy distribution contained non-finite values.",
                                n_steps + 1,
                            )
                            if (
                                config.nonfinite_debug
                                and nonfinite_debug_logs
                                < config.nonfinite_debug_max_logs
                            ):
                                log_nonfinite_forward_debug(
                                    step=n_steps + 1,
                                    chunk_idx=t // trunc_t,
                                    obs_chunk=obs_chunk,
                                    actions_chunk=actions_chunk,
                                    firsts_chunk=firsts_chunk,
                                    pi_distribution=pi_distribution,
                                    fabric=fabric,
                                )
                                nonfinite_debug_logs += 1
                        agent_state = None
                        optimizer.zero_grad(set_to_none=True)
                        pi_distribution = None
                        new_agent_state = None
                        maybe_clear_cuda_cache(device)
                        if skipped_nonfinite_updates > config.max_nonfinite_updates:
                            raise RuntimeError(
                                f"Exceeded max_nonfinite_updates={config.max_nonfinite_updates}"
                            )
                        continue

                    try:
                        log_prob = policy.get_logprob_of_action(
                            pi_distribution, actions_chunk
                        )
                    except AssertionError:
                        skipped_nonfinite_updates += 1
                        if fabric.is_global_zero:
                            logging.warning(
                                "Skipping step %s because policy log-prob contained NaNs.",
                                n_steps + 1,
                            )
                            if (
                                config.nonfinite_debug
                                and nonfinite_debug_logs
                                < config.nonfinite_debug_max_logs
                            ):
                                log_nonfinite_forward_debug(
                                    step=n_steps + 1,
                                    chunk_idx=t // trunc_t,
                                    obs_chunk=obs_chunk,
                                    actions_chunk=actions_chunk,
                                    firsts_chunk=firsts_chunk,
                                    pi_distribution=pi_distribution,
                                    fabric=fabric,
                                )
                                nonfinite_debug_logs += 1
                        agent_state = None
                        optimizer.zero_grad(set_to_none=True)
                        pi_distribution = None
                        new_agent_state = None
                        maybe_clear_cuda_cache(device)
                        if skipped_nonfinite_updates > config.max_nonfinite_updates:
                            raise RuntimeError(
                                f"Exceeded max_nonfinite_updates={config.max_nonfinite_updates}"
                            )
                        continue

                    loss_sum = -masked_sum(log_prob, valid_mask_chunk)
                    loss = loss_sum * fabric.world_size / global_valid_count.clamp_min(
                        1.0
                    )

                with timer.time("train backward"):
                    optimizer.zero_grad(set_to_none=True)
                    fabric.backward(loss)

                if skip_nonfinite_updates:
                    if not torch.isfinite(loss.detach()):
                        skipped_nonfinite_updates += 1
                        if fabric.is_global_zero:
                            logging.warning(
                                "Skipping step %s because loss is non-finite.",
                                n_steps + 1,
                            )
                            if (
                                config.nonfinite_debug
                                and nonfinite_debug_logs
                                < config.nonfinite_debug_max_logs
                            ):
                                log_nonfinite_forward_debug(
                                    step=n_steps + 1,
                                    chunk_idx=t // trunc_t,
                                    obs_chunk=obs_chunk,
                                    actions_chunk=actions_chunk,
                                    firsts_chunk=firsts_chunk,
                                    pi_distribution=pi_distribution,
                                    fabric=fabric,
                                )
                                for line in summarize_gradient_parameters(policy):
                                    logging.warning(line)
                                nonfinite_debug_logs += 1
                        agent_state = None
                        optimizer.zero_grad(set_to_none=True)
                        loss = None
                        log_prob = None
                        pi_distribution = None
                        new_agent_state = None
                        maybe_clear_cuda_cache(device)
                        if skipped_nonfinite_updates > config.max_nonfinite_updates:
                            raise RuntimeError(
                                f"Exceeded max_nonfinite_updates={config.max_nonfinite_updates}"
                            )
                        continue

                    nonfinite_grad_names = find_nonfinite_gradient_parameters(policy)
                    if nonfinite_grad_names:
                        skipped_nonfinite_updates += 1
                        if fabric.is_global_zero:
                            logging.warning(
                                "Skipping step %s due to non-finite gradients in: %s",
                                n_steps + 1,
                                ", ".join(nonfinite_grad_names),
                            )
                            if (
                                config.nonfinite_debug
                                and nonfinite_debug_logs
                                < config.nonfinite_debug_max_logs
                            ):
                                log_nonfinite_forward_debug(
                                    step=n_steps + 1,
                                    chunk_idx=t // trunc_t,
                                    obs_chunk=obs_chunk,
                                    actions_chunk=actions_chunk,
                                    firsts_chunk=firsts_chunk,
                                    pi_distribution=pi_distribution,
                                    fabric=fabric,
                                )
                                for line in summarize_gradient_parameters(policy):
                                    logging.warning(line)
                                nonfinite_debug_logs += 1
                        agent_state = None
                        optimizer.zero_grad(set_to_none=True)
                        loss = None
                        log_prob = None
                        pi_distribution = None
                        new_agent_state = None
                        maybe_clear_cuda_cache(device)
                        if skipped_nonfinite_updates > config.max_nonfinite_updates:
                            raise RuntimeError(
                                f"Exceeded max_nonfinite_updates={config.max_nonfinite_updates}"
                            )
                        continue

                with timer.time("train clip grad"):
                    grad_norm = fabric.clip_gradients(
                        policy, optimizer, max_norm=config.max_grad_norm
                        , error_if_nonfinite=False
                    )

                if skip_nonfinite_updates and grad_norm is not None:
                    if not torch.isfinite(grad_norm):
                        skipped_nonfinite_updates += 1
                        if fabric.is_global_zero:
                            logging.warning(
                                "Skipping step %s because clipped grad norm is non-finite.",
                                n_steps + 1,
                            )
                            if (
                                config.nonfinite_debug
                                and nonfinite_debug_logs
                                < config.nonfinite_debug_max_logs
                            ):
                                log_nonfinite_forward_debug(
                                    step=n_steps + 1,
                                    chunk_idx=t // trunc_t,
                                    obs_chunk=obs_chunk,
                                    actions_chunk=actions_chunk,
                                    firsts_chunk=firsts_chunk,
                                    pi_distribution=pi_distribution,
                                    fabric=fabric,
                                )
                                for line in summarize_gradient_parameters(policy):
                                    logging.warning(line)
                                nonfinite_debug_logs += 1
                        agent_state = None
                        optimizer.zero_grad(set_to_none=True)
                        loss = None
                        log_prob = None
                        pi_distribution = None
                        new_agent_state = None
                        maybe_clear_cuda_cache(device)
                        if skipped_nonfinite_updates > config.max_nonfinite_updates:
                            raise RuntimeError(
                                f"Exceeded max_nonfinite_updates={config.max_nonfinite_updates}"
                            )
                        continue

                with timer.time("train optimizer step"):
                    optimizer.step()

                agent_state = tree_map(lambda x: x.detach(), new_agent_state)

                n_steps += 1
                timer.throughput("valid_samples", global_valid_count_value)
                n_valid_samples_processed += global_valid_count_value
                n_sequence_slots_processed += sequence_slots_per_step
                reduced_loss_sum = fabric.all_reduce(loss_sum.detach(), reduce_op="sum")
                sum_loss += reduced_loss_sum.item()
                n_loss += global_valid_count_value

                # Log training metrics
                should_log = should_run_periodic(
                    n_steps, config.log_freq, config.log_first_step
                )
                should_validate = should_run_periodic(
                    n_steps, val_freq, config.validate_first_step
                )
                should_save = should_run_periodic(
                    n_steps, config.save_freq, config.save_first_step
                )

                if should_log or should_validate:
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
                            "processed_samples": n_valid_samples_processed,
                            "processed_sequence_slots": n_steps
                            * sequence_slots_per_step,
                            "epoch": n_steps / epoch_len,
                            "learning_rate": lr,
                            "grad_l2_norm": compute_gradient_l2_norm(policy),
                            "weight_l2_norm": compute_weight_l2_norm(policy),
                            "accuracy": accuracy_sum / global_valid_count.clamp_min(1.0),
                            "entropy": entropy_sum / global_valid_count.clamp_min(1.0),
                            "valid_fraction": global_valid_count.clamp_min(0.0)
                            / sequence_slots_per_step,
                            "skipped_nonfinite_updates": skipped_nonfinite_updates,
                        }
                    )
                    sum_loss, n_loss = 0, 0

                # Save model checkpoint
                if should_save:
                    save_last_checkpoint()

                if n_sequence_slots_processed >= next_snapshot_n_samples:
                    state_dict = policy.state_dict()
                    fabric.save(
                        os.path.join(checkpoint_dir, f"snapshot_{n_steps}.weights"),
                        state_dict,
                    )
                    next_snapshot_n_samples += config.snapshot_every_n_samples

                # Validation step
                if should_validate:
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

            if end_training:
                break

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
