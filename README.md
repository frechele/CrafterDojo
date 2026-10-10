<div align="center">

# CrafterDojo

**An Integrated Research Suite for Building Open-Ended Embodied Agents in Crafter**

Junyeong Park, Hyeonseo Cho, Sungjin Ahn

[![arXiv](https://img.shields.io/badge/arXiv-2508.13530-b31b1b.svg?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2508.13530) [![Project Page](https://img.shields.io/badge/Project-Page-4c8bf5.svg?logo=googlechrome&logoColor=white)](https://sites.google.com/view/crafterdojo) [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Weights-ffcc4d.svg)](https://huggingface.co/frechele/CrafterDojo)<br>
[![Dataset: Survival](https://img.shields.io/badge/Dataverse-CrafterPlay--Survival-a51c30.svg)](https://doi.org/10.7910/DVN/FHR2KS) [![Dataset: Gatherer](https://img.shields.io/badge/Dataverse-CrafterPlay--Gatherer-a51c30.svg)](https://doi.org/10.7910/DVN/JXNHRB) [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

CrafterDojo is an integrated research suite that turns Crafter into a lightweight, Minecraft-like testbed for general-purpose embodied agents. It provides foundation models, dataset toolkits and reference agents:

- **CrafterVPT** — behavior prior trained by behavior cloning on expert play (tiny / base / large)
- **CrafterCLIP** — video–language encoder for Crafter gameplay
- **CrafterSteve-1** — instruction-following agent built on CrafterVPT and CrafterCLIP
- **CrafterPlay / CrafterCaption** — toolkits for generating expert behavior data and rule-based captions
- **DEP / H-DEP** — planner-based agents that use CrafterSteve-1 as a low-level controller

## Installation

```bash
git clone https://github.com/frechele/CrafterDojo
cd CrafterDojo
uv sync --extra env --extra crafterclip
```

Extras: `env` (Craftax environment), `crafterclip` (CrafterCLIP / CrafterSteve-1), `dataset_generation`
(CrafterPlay / CrafterCaption), `preprocess` (Steve-1 preprocessing), `deps` (LLM planner), `wandb`.
Install only the ones you need, e.g. `uv sync --extra env` for CrafterVPT.

## Pretrained Models

Weights are hosted on the Hugging Face Hub at [`frechele/CrafterDojo`](https://huggingface.co/frechele/CrafterDojo) in `safetensors` format.

```bash
bash scripts/download_weights.sh          # everything below, into models/
bash scripts/download_weights.sh vpt      # or: clip | steve1
```

| Model | File | Architecture config | Size |
|---|---|---|---:|
| CrafterVPT-tiny | `models/vpt/tiny.safetensors` | `models/vpt/tiny.yaml` | 13 MB |
| CrafterVPT-base | `models/vpt/base.safetensors` | `models/vpt/base.yaml` | 63 MB |
| CrafterVPT-large | `models/vpt/large.safetensors` | `models/vpt/large.yaml` | 257 MB |
| CrafterCLIP | `models/crafterclip/crafterclip.safetensors` | — | 634 MB |
| CrafterSteve-1 | `models/steve1/steve1.safetensors` | `models/vpt/base_lora.yaml` | 65 MB |
| CrafterSteve-1 prior | `models/steve1/steve1_prior.safetensors` | — | 9 MB |

All models are trained on the `sv_gh_pp-10B_20ep` mixture of CrafterPlay data (Survival + Gatherer experts).

## Datasets

CrafterPlay episodes and CrafterCaption captions are hosted on Harvard Dataverse:

| Dataset | DOI |
|---|---|
| CrafterPlay-Survival (`survival-10B`) | [10.7910/DVN/FHR2KS](https://doi.org/10.7910/DVN/FHR2KS) |
| CrafterPlay-Gatherer (`gatherer-10B`) | [10.7910/DVN/JXNHRB](https://doi.org/10.7910/DVN/JXNHRB) |

Each dataset is split into `.tar.zst` shards. Extract them under `data/episodes/`:

```bash
mkdir -p data/episodes
tar --zstd -xf crafterdojo_playdata_raw_survival-10B_19967-19999_part-0231.tar.zst -C data/episodes
```

- [docs/dataset/DATASET.md](docs/dataset/DATASET.md) — statistics, file organization and data formats
- [docs/dataset/GENERATION.md](docs/dataset/GENERATION.md) — generating custom data with your own rewards and caption rules
- [notebooks/crafterdojo_tutorial.ipynb](notebooks/crafterdojo_tutorial.ipynb) — loading and visualizing episodes and captions
- [docs/dataset/croissant/](docs/dataset/croissant/) — Croissant metadata (with Responsible AI fields) for each dataset

## Quick Start

All training and evaluation goes through the Hydra entrypoint `main.py` (configs under `config/`).

**Evaluate CrafterVPT**

```bash
uv run python main.py mode=eval runner=vpt/bc env=craftax_pixels \
    runner.model=models/vpt/base.yaml runner.weights=models/vpt/base.safetensors
```

**Evaluate CrafterSteve-1 on an instruction**

```bash
uv run python main.py mode=eval runner=steve1/bc env=craftax_dojo env.task=harvest_sapling \
    "runner.eval.instruction=collect a sapling"
```

Tasks are defined in `src/crafterdojo/env/tasks/`. The Steve-1 eval config defaults to the released
weights and the conditioning scale used in the paper (`runner.eval.cond_scale=0.4`).

**Train**

```bash
bash scripts/training/train_vpt_bc.sh
bash scripts/training/train_crafterclip.sh     # needs: bash scripts/download_weights.sh openai
bash scripts/training/train_steve1_bc.sh
```

See [scripts/README.md](scripts/README.md) for defaults and environment-variable overrides.

**Generate data**

```bash
bash scripts/data/expert_train.sh survival 1000000
bash scripts/data/expert_generate.sh outputs/expert_train/policy.ckpt data/episodes 16 0
bash scripts/data/expert_process.sh data/episodes/Craftax-Classic-Symbolic-v1 16 0
bash scripts/data/expert_filter.sh data/episodes/Craftax-Classic-Symbolic-v1 16 0 20
```

The same steps are available as console scripts (`uv run expert-train --help`, `caption-generate`, …).
See [dataset_generation/README.md](dataset_generation/README.md) and [docs/dataset/GENERATION.md](docs/dataset/GENERATION.md).

## Repository Layout

```text
config/              Hydra configs for runners, environments and training
data/                Small caption / prompt resources used by evaluators and planners
dataset_generation/  CrafterPlay (expert behavior) and CrafterCaption toolkits
docs/dataset/        Dataset documentation
models/              Architecture configs; downloaded weights land here
notebooks/           Dataset tutorial
scripts/             Training, data-generation and weight-download scripts
src/crafterdojo/     Models, datasets, agents, environment wrappers, runners, evaluators
main.py              Hydra entrypoint
```

## Citation

```bibtex
@article{park2026crafterdojo,
  title   = {CrafterDojo: An Integrated Research Suite for Building Open-Ended Embodied Agents in Crafter},
  author  = {Park, Junyeong and Cho, Hyeonseo and Ahn, Sungjin},
  journal = {arXiv preprint arXiv:2508.13530},
  year    = {2026}
}
```

## License

Code is released under the [MIT License](LICENSE), which also lists third-party notices
(Craftax, OpenAI VPT, MineCLIP, OpenAI CLIP, JAX) and the STEVE-1 attribution. The released weights are distributed under the same license.
