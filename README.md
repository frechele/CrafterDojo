# CrafterDojo

Author: Junyeong Park, Hyeonseo Cho, Sungjin Ahn

[[paper]](https://arxiv.org/abs/2508.13530)

## Abstract

Developing general-purpose embodied agents is a core challenge in AI. Minecraft provides rich complexity and internet-scale data, but its slow speed and engineering overhead make it unsuitable for rapid prototyping. Crafter offers a lightweight alternative that retains key challenges from Minecraft, yet its use has remained limited to narrow tasks due to the absence of foundation models that have driven progress in the Minecraft setting. In this paper, we present CrafterDojo, a suite of foundation models and tools that unlock the Crafter environment as a lightweight, prototyping-friendly, and Minecraft-like testbed for general-purpose embodied agent research. CrafterDojo addresses this by introducing CrafterVPT, CrafterCLIP, and CrafterSteve-1 for behavior priors, vision-language grounding, and instruction following, respectively. In addition, we provide toolkits for generating behavior and caption datasets (CrafterPlay and CrafterCaption), reference agent implementations, benchmark evaluations, and a complete open-source codebase.


## Key Components

### Foundation Models

- **CrafterCLIP**: Vision-language model for understanding Crafter gameplay
- **CrafterVPT**: Video pre-training models (tiny, base, large variants) with LoRA support  
- **CrafterSteve-1**: Advanced agent model with hierarchical control

### Data Generator Toolkits

- **Expert Behavior Generator**: PPO-based expert policy training and demonstration generation
- **Caption Generator**: Rule-based system for creating descriptive captions from gameplay

## Installation

**Step 1. Clone the repository**
```bash
git clone https://github.com/frechele/CrafterDojo
cd CrafterDojo
```

**Step 2. Download pre-trained models**
```bash
uv run bash scripts/download_weights.sh
```

## Quick Start

### Training Agents

**Train VPT behavioral cloning**:
```bash
uv run bash scripts/train/vpt/run_bc.sh base  # train base model
```

**Train CrafterCLIP model**:
```bash
uv run bash scripts/train/run_cclip.sh
```

**Train CrafterSteve-1 model**:
```bash
uv run bash scripts/train/steve1/run_preprocess.sh
uv run bash scripts/train/steve1/run_steve1.sh
```

**Train PPO-Steve agent**:
```bash
uv run bash scripts/train/run_ppo_steve.sh harvest_sapling 10  # train for harvest sapling task with low-level steps=10
```


### Evaluation

```bash
uv run agent_main.py mode=eval env=crafterdojo env.task=harvest_sapling agent=ppo_steve eval.load=path/to/checkpoint
uv run scripts/gather_crafterdojo.py path/to/output
```

### Data Generation

- **Generate expert demonstrations**: Refer [README.md](toolkit/expert/README.md)
- **Generate captions**: Refer [README.md](toolkit/caption/README.md)

## Project Structure

```
CrafterDojo/
├── agent_main.py          # Main agent training/evaluation script
├── model_main.py          # Main model training script  
├── crafterdojo/           # Core package
│   ├── agent/             # Agent implementations
│   ├── model/             # Model architectures
│   ├── env/               # Environment wrappers
│   ├── data/              # Data loading utilities
│   └── common/            # Shared utilities
├── config/                # Hydra configurations
├── scripts/               # Training/utility scripts
├── toolkit/               # Data generation tools
│   ├── caption/           # Caption generation
│   └── expert/            # Expert behavior generation
└── models/                # Pre-trained model weights
```
