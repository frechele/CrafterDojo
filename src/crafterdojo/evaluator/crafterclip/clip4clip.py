"""CLIP4Clip video-text retrieval evaluator."""

import torch
from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)


CLIP_IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)


def preprocess_image(frames: torch.Tensor) -> torch.Tensor:
    mu = torch.tensor(CLIP_IMAGE_MEAN, device=frames.device)
    std = torch.tensor(CLIP_IMAGE_STD, device=frames.device)
    return (frames - mu[None, None, :, None, None]) / std[None, None, :, None, None]


class CLIP4ClipEvaluator:
    def __init__(self, model_name: str = "Searchium-ai/clip4clip-webvid150k", device: str = "cuda"):
        self.device = device
        self.model = CLIPVisionModelWithProjection.from_pretrained(model_name).to(device)
        self.text_model = CLIPTextModelWithProjection.from_pretrained(model_name).to(device)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.model.eval()
        self.text_model.eval()

    def extract_video_features(self, video_frames):
        b, t = video_frames.shape[:2]
        frames = preprocess_image(video_frames.float() / 255.0)
        frames = frames.view(b * t, *frames.shape[2:])
        with torch.no_grad():
            features = self.model(frames, interpolate_pos_encoding=True)["image_embeds"]
        features = features.view(b, t, -1)
        features = features / features.norm(dim=-1, keepdim=True)
        video_features = torch.mean(features, dim=1)
        return video_features / video_features.norm(dim=-1, keepdim=True)

    def extract_text_features(self, texts):
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=False)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.text_model(**inputs)
        return outputs[0] / outputs[0].norm(dim=-1, keepdim=True)

    def compute_similarity_matrix(self, video_features, text_embeddings):
        return torch.matmul(video_features, text_embeddings.T)
