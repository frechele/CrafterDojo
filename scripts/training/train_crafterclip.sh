#!/usr/bin/env bash
# CrafterCLIP training.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_DIR}"

DATASET_ROOT="${DATASET_ROOT:-data/episodes/playdata_raw/survival-10B/Craftax-Classic-Symbolic-v1}"
CAPTION_PATH="${CAPTION_PATH:-caption}"
TRAIN_SPLIT="${TRAIN_SPLIT:-[0:2]}"
VAL_SPLIT="${VAL_SPLIT:-[2:3]}"
REPHRASE_DICT="${REPHRASE_DICT:-null}"
NUM_REPHRASED="${NUM_REPHRASED:-0}"
OAI_CLIP_WEIGHTS="${OAI_CLIP_WEIGHTS:-models/clip/ViT-B-16.pt}"

DEVICES="${DEVICES:-1}"
ACCELERATOR="${ACCELERATOR:-auto}"
PRECISION="${PRECISION:-bf16-mixed}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-0}"
N_EPOCHS="${N_EPOCHS:-1}"
VAL_FREQ="${VAL_FREQ:-0}"
SAVE_FREQ="${SAVE_FREQ:-0}"
WANDB_ENABLE="${WANDB_ENABLE:-false}"
RUN_NAME="${RUN_NAME:-crafterclip_local}"
HYDRA_DIR="${HYDRA_DIR:-outputs/crafterclip/${RUN_NAME}}"

if [[ ! -f "${OAI_CLIP_WEIGHTS}" ]]; then
    echo "Missing OAI_CLIP_WEIGHTS: ${OAI_CLIP_WEIGHTS}" >&2
    exit 1
fi
if [[ "${REPHRASE_DICT}" != "null" && ! -f "${REPHRASE_DICT}" ]]; then
    echo "Missing REPHRASE_DICT: ${REPHRASE_DICT}" >&2
    exit 1
fi
if [[ ! -d "${DATASET_ROOT}/video" || ! -d "${DATASET_ROOT}/${CAPTION_PATH}" || ! -d "${DATASET_ROOT}/len" ]]; then
    echo "DATASET_ROOT must contain video/, ${CAPTION_PATH}/, and len/: ${DATASET_ROOT}" >&2
    exit 1
fi

uv run --extra crafterclip python main.py \
    runner=crafterclip/default \
    "runner.oai_clip_weights=${OAI_CLIP_WEIGHTS}" \
    "runner.rephrase_dict=${REPHRASE_DICT}" \
    "runner.num_rephrased=${NUM_REPHRASED}" \
    "runner.train_dataset.dataset_root=${DATASET_ROOT}" \
    "runner.train_dataset.caption_path=${CAPTION_PATH}" \
    "runner.train_dataset.data_split=\"${TRAIN_SPLIT}\"" \
    "runner.val_dataset.dataset_root=${DATASET_ROOT}" \
    "runner.val_dataset.caption_path=${CAPTION_PATH}" \
    "runner.val_dataset.data_split=\"${VAL_SPLIT}\"" \
    "runner.batch_size=${BATCH_SIZE}" \
    "runner.num_workers=${NUM_WORKERS}" \
    "runner.n_epochs=${N_EPOCHS}" \
    "runner.val_freq=${VAL_FREQ}" \
    "runner.save_freq=${SAVE_FREQ}" \
    "fabric.accelerator=${ACCELERATOR}" \
    "fabric.devices=${DEVICES}" \
    "fabric.precision=${PRECISION}" \
    "wandb.enable=${WANDB_ENABLE}" \
    "wandb.name=${RUN_NAME}" \
    "hydra.run.dir=${HYDRA_DIR}"
