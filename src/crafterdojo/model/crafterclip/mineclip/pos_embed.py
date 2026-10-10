import torch
import numpy as np
from einops import rearrange


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_1d_sincos_pos_embed(embed_dim, seq_len: int, cls_token=False):
    """
    Returns:
        [seq_len, embed_dim] or [1+seq_len, embed_dim] (w/ or w/o cls_token)
    """
    grid = np.arange(seq_len, dtype=np.float32)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return torch.tensor(pos_embed, dtype=torch.float)


def interpolate_resize_pos_embed(pos_embed, old_size, new_size):
    """
    NOTE: remove cls token from pos_embed first before passing it here

    Args:
        pos_embed: [seq_len, embed_dim]
        old_size: [h, w], seq_len of pos_embed must be equal to h * w
        new_size: [new_h, new_w]
    """
    old_hw, D = pos_embed.size()
    if isinstance(old_size, int):
        old_size = (old_size, old_size)
    if isinstance(new_size, int):
        new_size = (new_size, new_size)
    assert len(old_size) == 2
    assert len(new_size) == 2
    old_h, old_w = old_size
    assert old_h * old_w == old_hw
    pos_embed = rearrange(pos_embed, "(H W) D -> 1 D H W", H=old_h)
    new_embed = torch.nn.functional.interpolate(
        pos_embed, size=new_size, mode="bicubic", align_corners=False
    )
    new_embed = rearrange(new_embed, "1 D H W -> (H W) D")
    assert new_embed.size() == (new_size[0] * new_size[1], D)
    return new_embed


def infer_patch_grid(num_patches):
    best = None
    for h in range(1, int(num_patches**0.5) + 1):
        if num_patches % h != 0:
            continue
        w = num_patches // h
        if best is None or abs(h - w) < abs(best[0] - best[1]):
            best = (h, w)
    if best is None:
        raise ValueError(f"Unable to infer patch grid for {num_patches} patches.")
    return best


def resize_pos_embed_in_state_dict(state_dict, target_resolution, patch_size):
    pos_embed_key = None
    pos_embed = None
    for key, value in state_dict.items():
        if key.endswith("pos_embed") and value.ndim == 2 and value.shape[0] > 100:
            pos_embed_key = key
            pos_embed = value
            break
    if pos_embed is None:
        return state_dict

    num_patches = pos_embed.shape[0] - 1
    old_h, old_w = infer_patch_grid(num_patches)
    target_h = target_resolution[0] // patch_size
    target_w = target_resolution[1] // patch_size
    if (old_h, old_w) == (target_h, target_w):
        return state_dict

    cls_token, old_grid = pos_embed[:1], pos_embed[1:]
    new_grid = interpolate_resize_pos_embed(old_grid, (old_h, old_w), (target_h, target_w))
    resized = torch.cat([cls_token, new_grid], dim=0)

    for key, value in state_dict.items():
        if key.endswith("pos_embed") and value.ndim == 2 and value.shape[0] == num_patches + 1:
            state_dict[key] = resized

    return state_dict
