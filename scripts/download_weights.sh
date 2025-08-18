#! /bin/bash

# To train CrafterCLIP, uncomment the following line:
# mkdir -p models/clip
# wget -O models/clip/ViT-B-16.pt https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt
# python crafterdojo/model/crafterclip/convert_clip.py

# Pre-trained CrafterCLIP model
mkdir -p models/crafterclip
gdown -O models/crafterclip/cclip.weights https://drive.google.com/uc?id=1NQHg_4Udc34mib3-cQaPhzrIr8w-Oc1W


# Pre-trained CrafterVPT models
mkdir -p models/craftervpt
gdown -O models/craftervpt/cvpt_tiny.weights https://drive.google.com/uc?id=1ZLGj0GZc-cmCv5hifCN1FvYNzYj6p0ZY
gdown -O models/craftervpt/cvpt_tiny.model https://drive.google.com/uc?id=1EuU2UAlX2DOFYm9VwWSng1z9QUaJ37HK
gdown -O models/craftervpt/cvpt_base.weights https://drive.google.com/uc?id=1qmqFLJxThc7v9HTT5oL8hsxNwJ0UFmMz
gdown -O models/craftervpt/cvpt_base.model https://drive.google.com/uc?id=1pRUYvS0EztfztHYNzXzOIuDDxJAjCmFb
gdown -O models/craftervpt/cvpt_base_lora.model https://drive.google.com/uc?id=15SGwVv4i_rUIHUFPlKL-3WaFONWCjqAH
gdown -O models/craftervpt/cvpt_large.weights https://drive.google.com/uc?id=1QZkP_YrMQdUKnaCrH3ysP_WX1hOTFIem
gdown -O models/craftervpt/cvpt_large.model https://drive.google.com/uc?id=1scPpA9dKL9RmjMBA5vZ6z2Cs5XMeF9qm


# Pre-trained CrafterSteve-1 model
mkdir -p models/craftersteve1
gdown -O models/craftersteve1/csteve1.weights https://drive.google.com/uc?id=1rtYRw4prJLZUiDLALkUwqeyasIqdf401
gdown -O models/craftersteve1/csteve1_prior.pt https://drive.google.com/uc?id=1tiWV_Xm4-0CEYbF5zhSfoIcZSEujlQG2
