#! /bin/bash

python -m src.train \
    --env_name "Craftax-Classic-Symbolic-v1" \
    --seed 1234 --save_policy \
    --total_timesteps 1e9 \
    --num_envs 4096 \
    --death_penalty 10
