#! /bin/bash

MODELNAME=$1

python agent_main.py mode=train agent=vpt/bc fabric.devices=1 \
    agent.model=models/craftervpt/cvpt_${MODELNAME}.model \
    agent.learning_rate=1e-4 agent.batch_size=32
