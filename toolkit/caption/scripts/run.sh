#! /bin/bash

python generate_caption.py \
    ../../CrafterDojo_Data

python caption_uniform_sampling.py \
    ../../CrafterDojo_Data/caption \
    statistics_*.pkl
