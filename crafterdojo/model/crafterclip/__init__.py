from crafterdojo.model.crafterclip.mineclip import MineCLIP


CRAFTER_MINECLIP_CONFIG = {
    "arch": "vit_base_p16_fz.v2.t2",
    "hidden_dim": 512,
    "image_feature_dim": 512,
    "mlp_adapter_spec": "v0-2.t0",
    "pool_type": "attn.d2.nh8.glusw",
    "resolution": [144, 144],
    "ckpt": "models/crafterclip/cclip.weights", 
}


def load_mineclip_wconfig(device, cfg = None) -> MineCLIP:
    if cfg is None:
        cfg = CRAFTER_MINECLIP_CONFIG.copy()
    ckpt = cfg.pop("ckpt")

    model = MineCLIP(**cfg).to(device)
    model.load_ckpt(ckpt, strip_prefix=None, strict=True)
    return model
