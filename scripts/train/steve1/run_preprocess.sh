#! /bin/bash

# Extract CLIP embeddings
python -m crafterdojo.agent.steve1.preprocess.extract_clip_embedding \
    CrafterDojo_Data/filtered_20


# Extract Event Boundaries
python -m crafterdojo.agent.steve1.preprocess.event_bound \
    CrafterDojo_Data \
    CrafterDojo_Data/filtered_20


# Extract CVAE training dataset
python -m crafterdojo.agent.steve1.preprocess.extract_text_embedding \
    CrafterDojo_Data


# Train CVAE
python -m crafterdojo.agent.steve1.train_vae \
    --data_path CrafterDojo_Data/text_embeddings.npz
