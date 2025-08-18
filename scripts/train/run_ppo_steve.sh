#! /bin/bash

task=$1
lls=$2

python agent_main.py mode=train env=crafterdojo env.task=$task agent=ppo_steve agent.low_level_steps=$lls
