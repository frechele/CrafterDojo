#!/usr/bin/env bash
# Train a Craftax expert policy used for CrafterPlay generation.

set -euo pipefail

REWARD_WRAPPER="${1:-survival}"
TOTAL_TIMESTEPS="${2:-1000000}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_DIR}"

NUM_ENVS="${NUM_ENVS:-4096}"
SEED="${SEED:-42}"
USE_WANDB="${USE_WANDB:-false}"
WANDB_PROJECT="${WANDB_PROJECT:-crafterdojo-playdata}"
WANDB_FLAG="--no-use_wandb"
if [[ "${USE_WANDB}" == "true" ]]; then
    WANDB_FLAG="--use_wandb"
fi

uv run --extra dataset_generation expert-train \
    --reward_wrappers "${REWARD_WRAPPER}" \
    --total_timesteps "${TOTAL_TIMESTEPS}" \
    --num_envs "${NUM_ENVS}" \
    --seed "${SEED}" \
    --save_policy \
    --wandb_project "${WANDB_PROJECT}" \
    "${WANDB_FLAG}"
