"""OpenAI CLIP (MineCLIP-backbone `CLIP`) video-text retrieval evaluator."""

from pathlib import Path

import torch

from crafterdojo.model.crafterclip.mineclip.clip import CLIP
from crafterdojo.model.crafterclip.utils.image_utils import basic_image_tensor_preprocess

from .loaders import (
    OPENAI_CLIP_PATCH_SIZE,
    load_openai_clip_state_dict,
    resolve_openai_weights_path,
)


CLIP_IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)


class OpenAICLIPEvaluator:
    def __init__(
        self,
        device: str = "cuda",
        weights_path: str | None = None,
        image_resolution: tuple[int, int] = (144, 144),
    ):
        self.device = device
        if isinstance(image_resolution, (tuple, list)):
            image_resolution = (int(image_resolution[0]), int(image_resolution[1]))
        else:
            image_resolution = (int(image_resolution), int(image_resolution))
        self.image_resolution = image_resolution

        clip_config = {
            "context_length": 77,
            "embed_dim": 512,
            "image_resolution": self.image_resolution[0],
            "text_heads": 8,
            "text_layers": 12,
            "text_width": 512,
            "vision_layers": 12,
            "vision_patch_size": OPENAI_CLIP_PATCH_SIZE,
            "vision_width": 768,
            "vocab_size": 49408,
        }
        self.model = CLIP(**clip_config)
        repo_root = Path(__file__).resolve().parents[4]
        weights_path = resolve_openai_weights_path(weights_path, repo_root=repo_root)
        state_dict = load_openai_clip_state_dict(weights_path, self.image_resolution)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(device).eval()
        self.weights_path = weights_path

    def extract_video_features(self, video_frames):
        b, t = video_frames.shape[:2]
        frames = basic_image_tensor_preprocess(
            video_frames,
            mean=CLIP_IMAGE_MEAN,
            std=CLIP_IMAGE_STD,
            shape=self.image_resolution,
        )
        frames = frames.view(b * t, *frames.shape[2:])
        with torch.no_grad():
            features = self.model.encode_image(frames)
        features = features.view(b, t, -1)
        features = features / features.norm(dim=-1, keepdim=True)
        video_features = torch.mean(features, dim=1)
        return video_features / video_features.norm(dim=-1, keepdim=True)

    def extract_text_features(self, texts):
        text_embeddings = self.model.encode_text(texts)
        return text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)

    def compute_similarity_matrix(self, video_features, text_embeddings):
        return torch.matmul(video_features, text_embeddings.T)
