#!/usr/bin/env bash
# Roll out a trained expert policy and write symbolic episode files.

set -euo pipefail

CHECKPOINT_PATH="${1:?usage: expert_generate.sh <checkpoint_path> <save_path> [n_episodes] [start_ep]}"
SAVE_PATH="${2:?usage: expert_generate.sh <checkpoint_path> <save_path> [n_episodes] [start_ep]}"
N_EPISODES="${3:-20000}"
START_EP="${4:-0}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_DIR}"

MAX_STEPS="${MAX_STEPS:-10000}"
SEED="${SEED:-42}"

if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
    echo "Missing checkpoint: ${CHECKPOINT_PATH}" >&2
    exit 1
fi
mkdir -p "${SAVE_PATH}"

uv run --extra dataset_generation expert-generate \
    --checkpoint_path "${CHECKPOINT_PATH}" \
    --save_path "${SAVE_PATH}" \
    --n_episodes "${N_EPISODES}" \
    --start_ep "${START_EP}" \
    --max_steps "${MAX_STEPS}" \
    --seed "${SEED}"
