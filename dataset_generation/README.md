# Dataset Generation

`dataset_generation` now has two explicit entrypoints:

- `dataset_generation/expert_behavior/main.py`: expert behavior pipeline
- `dataset_generation/caption/main.py`: caption generation pipeline

## Structure

```text
dataset_generation/
├── main.py
├── expert_behavior/
│   ├── main.py
│   ├── train.py
│   ├── generate.py
│   ├── process.py
│   ├── filter.py
│   ├── nn/
│   ├── rewards/
│   └── wrappers/
└── caption/
    ├── main.py
    ├── sampling.py
    ├── gather.py
    └── rules/
```

## Which File Is The Main One?

- Expert behavior generation: `dataset_generation/expert_behavior/main.py`
- Caption generation: `dataset_generation/caption/main.py`
- Top-level dispatcher: `dataset_generation/main.py`

## Expert Behavior Pipeline

`train.py` trains the policy.

`generate.py` rolls out the trained policy and writes symbolic `state/*.npz`.

`process.py` converts symbolic episodes into `video/`, `action/`, and `len/`.

`filter.py` removes long null-action segments from processed episodes.

## Caption Pipeline

`main.py` reads `state/*.npz` and writes caption annotations.

`sampling.py` balances caption frequency.

`gather.py` collects the global caption vocabulary.
