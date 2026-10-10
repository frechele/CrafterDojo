"""MineCLIP video-text retrieval evaluator."""

import torch
import torch.nn.functional as F

from crafterdojo.model.crafterclip import CRAFTER_MINECLIP_CONFIG
from crafterdojo.model.crafterclip.mineclip import MineCLIP
from crafterdojo.model.crafterclip.mineclip.pos_embed import resize_pos_embed_in_state_dict


class MineCLIPEvaluator:
    def __init__(self, ckpt_path: str = "models/mineclip/attn_minecraft.pth", device: str = "cuda"):
        self.device = device
        cfg = CRAFTER_MINECLIP_CONFIG.copy()
        target_resolution = tuple(cfg["resolution"])
        cfg.pop("ckpt", None)

        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt.get("state_dict", {})
        state_dict = resize_pos_embed_in_state_dict(state_dict, target_resolution, patch_size=16)
        strip_prefix = "model." if any(k.startswith("model.") for k in state_dict) else None
        ckpt["state_dict"] = state_dict

        self.model = MineCLIP(**cfg).to(device)
        self.model.load_ckpt(ckpt, strip_prefix=strip_prefix, strict=True)
        self.model.eval()
        self.ckpt_path = ckpt_path

    def extract_video_features(self, video_frames):
        video_features = self.model.encode_video(video_frames)
        return F.normalize(video_features, dim=-1)

    def extract_text_features(self, texts):
        text_embeddings = self.model.encode_text(texts)
        return F.normalize(text_embeddings, dim=-1)

    def compute_similarity_matrix(self, video_features, text_embeddings):
        return torch.matmul(video_features, text_embeddings.T)
