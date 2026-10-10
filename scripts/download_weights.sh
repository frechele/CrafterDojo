#!/usr/bin/env bash
# Download the released CrafterDojo v1 weights from the Hugging Face Hub into models/.
#
#   bash scripts/download_weights.sh            # all released models
#   bash scripts/download_weights.sh vpt        # only CrafterVPT (tiny/base/large)
#   bash scripts/download_weights.sh clip       # only CrafterCLIP
#   bash scripts/download_weights.sh steve1     # CrafterSteve-1 + its prior
#   bash scripts/download_weights.sh openai     # OpenAI CLIP ViT-B/16 (only needed to train CrafterCLIP)

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

HF_REPO="${HF_REPO:-frechele/CrafterDojo}"
HF_REVISION="${HF_REVISION:-v1.0}"
OPENAI_CLIP_URL="https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt"

hf_download() {
    uv run hf download "${HF_REPO}" --revision "${HF_REVISION}" --local-dir models "$@"
}

target="${1:-all}"
case "${target}" in
    all)    hf_download --include "*.safetensors" ;;
    vpt)    hf_download --include "vpt/*.safetensors" ;;
    clip)   hf_download --include "crafterclip/*.safetensors" ;;
    steve1) hf_download --include "steve1/*.safetensors" ;;
    openai)
        mkdir -p models/clip
        curl -L --fail -o models/clip/ViT-B-16.pt "${OPENAI_CLIP_URL}"
        ;;
    *) echo "unknown target: ${target} (all|vpt|clip|steve1|openai)" >&2; exit 1 ;;
esac
