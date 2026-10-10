#!/usr/bin/env bash
# CrafterSteve-1 goal-conditioned behavior cloning.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_DIR}"

DATASET_ROOT="${DATASET_ROOT:-data/episodes/playdata_raw/survival-10B/Craftax-Classic-Symbolic-v1}"
TRAIN_SPLIT="${TRAIN_SPLIT:-[0:2]}"
VAL_SPLIT="${VAL_SPLIT:-[2:3]}"
MODEL_PATH="${MODEL_PATH:-models/vpt/base_lora.yaml}"
INIT_WEIGHTS="${INIT_WEIGHTS:-}"
REPHRASED="${REPHRASED:-39}"
EMBED_TYPE="${EMBED_TYPE:-clip}"
COND_POSITION="${COND_POSITION:-output}"
SURPRISE_WEIGHT="${SURPRISE_WEIGHT:-true}"
SURPRISE_WEIGHT_CLIP="${SURPRISE_WEIGHT_CLIP:-2.83}"

DEVICES="${DEVICES:-1}"
ACCELERATOR="${ACCELERATOR:-auto}"
PRECISION="${PRECISION:-bf16-mixed}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-0}"
N_SAMPLES="${N_SAMPLES:-4096}"
N_EPOCHS="${N_EPOCHS:-0}"
T="${T:-64}"
TRUNC_T="${TRUNC_T:-64}"
VAL_FREQ="${VAL_FREQ:-0}"
SAVE_FREQ="${SAVE_FREQ:-0}"
WANDB_ENABLE="${WANDB_ENABLE:-false}"
RUN_NAME="${RUN_NAME:-steve1_bc_local}"
HYDRA_DIR="${HYDRA_DIR:-outputs/steve1_bc/${RUN_NAME}}"

EXTRA_ARGS=()
if [[ -n "${INIT_WEIGHTS}" ]]; then
    if [[ ! -f "${INIT_WEIGHTS}" ]]; then
        echo "Missing INIT_WEIGHTS: ${INIT_WEIGHTS}" >&2
        exit 1
    fi
    EXTRA_ARGS+=("runner.weights=${INIT_WEIGHTS}")
fi

if [[ ! -f "${MODEL_PATH}" ]]; then
    echo "Missing MODEL_PATH: ${MODEL_PATH}" >&2
    exit 1
fi
if [[ ! -d "${DATASET_ROOT}/video" || ! -d "${DATASET_ROOT}/action" || ! -d "${DATASET_ROOT}/len" ]]; then
    echo "DATASET_ROOT must contain video/, action/, and len/: ${DATASET_ROOT}" >&2
    exit 1
fi
case "${EMBED_TYPE}" in
    sbert) EMBEDDING_DIR="${DATASET_ROOT}/sbert_embedding" ;;
    clip_ach) EMBEDDING_DIR="${DATASET_ROOT}/clip_ach_embedding" ;;
    *) EMBEDDING_DIR="${DATASET_ROOT}/clip_embedding_reph${REPHRASED}" ;;
esac
if [[ ! -d "${EMBEDDING_DIR}" ]]; then
    echo "Missing embedding directory for EMBED_TYPE=${EMBED_TYPE}: ${EMBEDDING_DIR}" >&2
    exit 1
fi
if [[ ! -d "${DATASET_ROOT}/event" ]]; then
    echo "Missing event directory: ${DATASET_ROOT}/event" >&2
    exit 1
fi

uv run python main.py \
    runner=steve1/bc \
    "runner.model=${MODEL_PATH}" \
    "${EXTRA_ARGS[@]}" \
    "runner.train_dataset.dataset_root=${DATASET_ROOT}" \
    "runner.train_dataset.data_split=\"${TRAIN_SPLIT}\"" \
    "runner.val_dataset.dataset_root=${DATASET_ROOT}" \
    "runner.val_dataset.data_split=\"${VAL_SPLIT}\"" \
    "runner.rephrased=${REPHRASED}" \
    "runner.embed_type=${EMBED_TYPE}" \
    "runner.cond_position=${COND_POSITION}" \
    "runner.surprise_weight=${SURPRISE_WEIGHT}" \
    "runner.surprise_weight_clip=${SURPRISE_WEIGHT_CLIP}" \
    "runner.T=${T}" \
    "runner.trunc_t=${TRUNC_T}" \
    "runner.batch_size=${BATCH_SIZE}" \
    "runner.num_workers=${NUM_WORKERS}" \
    "runner.n_samples=${N_SAMPLES}" \
    "runner.n_epochs=${N_EPOCHS}" \
    "runner.val_freq=${VAL_FREQ}" \
    "runner.save_freq=${SAVE_FREQ}" \
    "fabric.accelerator=${ACCELERATOR}" \
    "fabric.devices=${DEVICES}" \
    "fabric.precision=${PRECISION}" \
    "wandb.enable=${WANDB_ENABLE}" \
    "wandb.name=${RUN_NAME}" \
    "hydra.run.dir=${HYDRA_DIR}"
