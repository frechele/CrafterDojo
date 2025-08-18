import torch
from collections import OrderedDict
from einops import rearrange


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


def main():
    state_dict = torch.jit.load("models/clip/ViT-B-16.pt").state_dict()

    print({k.split('.')[0] for k in state_dict.keys()})
    print({k: v.shape for k, v in state_dict.items() if k.startswith("visual")})

    new_state_dict = OrderedDict()

    # common
    new_state_dict["logit_scale"] = state_dict["logit_scale"]

    # vision_model
    new_state_dict["vision_model.cls_token"] = state_dict["visual.class_embedding"]
    new_state_dict["vision_model.projection"] = state_dict["visual.proj"]
    new_state_dict["vision_model.conv1.weight"] = state_dict["visual.conv1.weight"]
    new_state_dict["vision_model.ln_pre.weight"] = state_dict["visual.ln_pre.weight"]
    new_state_dict["vision_model.ln_pre.bias"] = state_dict["visual.ln_pre.bias"]
    new_state_dict["vision_model.ln_post.weight"] = state_dict["visual.ln_post.weight"]
    new_state_dict["vision_model.ln_post.bias"] = state_dict["visual.ln_post.bias"]

    old_embed = state_dict["visual.positional_embedding"]
    cls_embed, old_embed = old_embed[:1], old_embed[1:]
    new_embed = interpolate_resize_pos_embed(
        old_embed,
        224 // 16,
        [r // 16 for r in [144, 144]]
    )
    new_state_dict["vision_model.pos_embed"] = torch.cat([cls_embed, new_embed], dim=0)

    for orig_k, v in {k: v for k, v in state_dict.items() if k.startswith("visual.transformer.resblocks")}.items():
        new_k = orig_k.replace("visual.transformer.resblocks", "vision_model.blocks")
        new_state_dict[new_k] = v

    # text_model
    new_state_dict["text_model.token_embedding.weight"] = state_dict["token_embedding.weight"]
    new_state_dict["text_model.pos_embed"] = state_dict["positional_embedding"]
    new_state_dict["text_model.projection"] = state_dict["text_projection"]
    new_state_dict["text_model.ln_final.weight"] = state_dict["ln_final.weight"]
    new_state_dict["text_model.ln_final.bias"] = state_dict["ln_final.bias"]

    for orig_k, v in {k: v for k, v in state_dict.items() if k.startswith("transformer")}.items():
        new_k = orig_k.replace("transformer.resblocks", "text_model.blocks")
        new_state_dict[new_k] = v

    # let's save the new state dict!
    torch.save(new_state_dict, "models/clip/ViT-B-16.weights")


if __name__ == "__main__":
    main()
