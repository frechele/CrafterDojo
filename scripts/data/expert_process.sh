#!/usr/bin/env bash
# Convert symbolic episodes into video/action/length files.

set -euo pipefail

ROOT="${1:?usage: expert_process.sh <dataset_root> [n_episodes] [start_ep]}"
N_EPISODES="${2:-20000}"
START_EP="${3:-0}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_DIR}"

SAVE_WORKERS="${SAVE_WORKERS:-4}"
LOAD_WORKERS="${LOAD_WORKERS:-4}"

if [[ ! -d "${ROOT}/state" ]]; then
    echo "ROOT must contain state/: ${ROOT}" >&2
    exit 1
fi

uv run --extra dataset_generation expert-process \
    --root "${ROOT}" \
    --n_episodes "${N_EPISODES}" \
    --start_ep "${START_EP}" \
    --save_workers "${SAVE_WORKERS}" \
    --load_workers "${LOAD_WORKERS}"
