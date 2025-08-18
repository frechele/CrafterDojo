#! /bin/bash

task=$1

python agent_main.py mode=train env=crafterdojo env.task=$task agent=ppo
