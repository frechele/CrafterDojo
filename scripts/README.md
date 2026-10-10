# Scripts

These scripts provide local launch commands for the training and data-generation workflows.

All scripts are intended to be run from the repository root or from any directory.

Training scripts:

```bash
bash scripts/training/train_vpt_bc.sh
bash scripts/training/train_crafterclip.sh
bash scripts/training/train_steve1_bc.sh
```

Dataset generation helpers:

```bash
bash scripts/data/expert_train.sh survival 1000000
bash scripts/data/expert_generate.sh outputs/expert_train/policy.ckpt data/episodes 16 0
bash scripts/data/expert_process.sh data/episodes/Craftax-Classic-Symbolic-v1 16 0
bash scripts/data/expert_filter.sh data/episodes/Craftax-Classic-Symbolic-v1 16 0 20
```

Pretrained weights (Hugging Face Hub, `frechele/CrafterDojo`):

```bash
bash scripts/download_weights.sh            # all released models into models/
bash scripts/download_weights.sh openai     # OpenAI CLIP ViT-B/16, only needed to train CrafterCLIP
```

Dataset documentation helpers (regenerate the statistics and figures in `docs/dataset/`):

```bash
python scripts/dataset_stats.py --help
python scripts/dataset_stats_figure.py
CRAFTERDOJO_SAMPLE_ROOT=<sample_dir> python scripts/dataset_anatomy_figure.py
```
