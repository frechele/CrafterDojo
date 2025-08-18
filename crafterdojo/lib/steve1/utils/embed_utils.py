import torch


def get_prior_embed(text, mineclip, prior, device):
    """Get the embed processed by the prior."""
    text_embed = mineclip.encode_text(text).detach().cpu().numpy()
    text_prompt_embed = prior(torch.tensor(text_embed).to(device)).cpu().detach().numpy()
    return text_prompt_embed
