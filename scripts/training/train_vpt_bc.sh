#!/usr/bin/env bash
# CrafterVPT behavior cloning training.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_DIR}"

DATASET_ROOT="${DATASET_ROOT:-data/episodes/playdata_raw/survival-10B/Craftax-Classic-Symbolic-v1}"
TRAIN_SPLIT="${TRAIN_SPLIT:-[0:2]}"
VAL_SPLIT="${VAL_SPLIT:-[2:3]}"
MODEL_PATH="${MODEL_PATH:-models/vpt/tiny.yaml}"
WEIGHTS_PATH="${WEIGHTS_PATH:-}"

DEVICES="${DEVICES:-1}"
ACCELERATOR="${ACCELERATOR:-cpu}"
PRECISION="${PRECISION:-null}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-0}"
N_EPOCHS="${N_EPOCHS:-1}"
N_SAMPLES="${N_SAMPLES:-1000}"
T="${T:-64}"
TRUNC_T="${TRUNC_T:-64}"
VAL_FREQ="${VAL_FREQ:-0}"
SAVE_FREQ="${SAVE_FREQ:-0}"
WANDB_ENABLE="${WANDB_ENABLE:-false}"
RUN_NAME="${RUN_NAME:-vpt_bc_local}"
HYDRA_DIR="${HYDRA_DIR:-outputs/vpt_bc/${RUN_NAME}}"

EXTRA_ARGS=()
if [[ -n "${WEIGHTS_PATH}" ]]; then
    if [[ ! -f "${WEIGHTS_PATH}" ]]; then
        echo "Missing WEIGHTS_PATH: ${WEIGHTS_PATH}" >&2
        exit 1
    fi
    EXTRA_ARGS+=("runner.weights=${WEIGHTS_PATH}")
fi

if [[ ! -f "${MODEL_PATH}" ]]; then
    echo "Missing MODEL_PATH: ${MODEL_PATH}" >&2
    exit 1
fi
if [[ ! -d "${DATASET_ROOT}/video" || ! -d "${DATASET_ROOT}/action" || ! -d "${DATASET_ROOT}/len" ]]; then
    echo "DATASET_ROOT must contain video/, action/, and len/: ${DATASET_ROOT}" >&2
    exit 1
fi

uv run python main.py \
    runner=vpt/bc \
    "runner.model=${MODEL_PATH}" \
    "${EXTRA_ARGS[@]}" \
    "runner.train_dataset.dataset_root=${DATASET_ROOT}" \
    "runner.train_dataset.data_split=\"${TRAIN_SPLIT}\"" \
    "runner.val_dataset.dataset_root=${DATASET_ROOT}" \
    "runner.val_dataset.data_split=\"${VAL_SPLIT}\"" \
    "runner.T=${T}" \
    "runner.trunc_t=${TRUNC_T}" \
    "runner.batch_size=${BATCH_SIZE}" \
    "runner.num_workers=${NUM_WORKERS}" \
    "runner.n_epochs=${N_EPOCHS}" \
    "runner.n_samples=${N_SAMPLES}" \
    "runner.val_freq=${VAL_FREQ}" \
    "runner.save_freq=${SAVE_FREQ}" \
    runner.save_first_step=false \
    runner.validate_first_step=false \
    "fabric.accelerator=${ACCELERATOR}" \
    "fabric.devices=${DEVICES}" \
    "fabric.precision=${PRECISION}" \
    "wandb.enable=${WANDB_ENABLE}" \
    "wandb.name=${RUN_NAME}" \
    "hydra.run.dir=${HYDRA_DIR}"
