import torch
import numpy as np


def get_prior_embed(text, mineclip, prior, device):
    """Get the embed processed by the prior."""
    text_embed = mineclip.encode_text(text).detach().cpu().numpy()
    text_prompt_embed = prior(torch.tensor(text_embed).to(device)).cpu().detach().numpy()
    return text_prompt_embed


def get_sbert_embed(text, sbert_model):
    """Get SentenceBERT text embedding directly. No VAE prior needed."""
    embed = sbert_model.encode(text, convert_to_numpy=True)
    if embed.ndim == 1:
        embed = embed.reshape(1, -1)
    return embed.astype(np.float32)
