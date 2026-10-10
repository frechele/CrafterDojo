#!/usr/bin/env bash
# Remove long null-action segments from processed episodes.

set -euo pipefail

ROOT="${1:?usage: expert_filter.sh <dataset_root> [n_episodes] [start_ep] [filter_len]}"
N_EPISODES="${2:-20000}"
START_EP="${3:-0}"
FILTER_LEN="${4:-20}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_DIR}"

if [[ ! -d "${ROOT}/video" || ! -d "${ROOT}/action" || ! -d "${ROOT}/len" ]]; then
    echo "ROOT must contain video/, action/, and len/: ${ROOT}" >&2
    exit 1
fi

uv run --extra dataset_generation expert-filter \
    --root "${ROOT}" \
    --filter "${FILTER_LEN}" \
    --n_episodes "${N_EPISODES}" \
    --start_ep "${START_EP}" \
    --overwrite
