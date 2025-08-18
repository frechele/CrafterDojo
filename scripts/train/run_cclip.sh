#! /bin/bash

python model_main.py mode=train model=crafterclip fabric.devices=8 \
    model.num_rephrased=39
