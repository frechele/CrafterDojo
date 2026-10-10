"""OpenCLIP-based video-text retrieval evaluator."""

import open_clip
import torch
from torchvision.transforms.functional import to_pil_image


class OpenCLIPEvaluator:
    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai", device: str = "cuda"):
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()
        image_size = getattr(self.model.visual, "image_size", 224)
        self.image_size = (image_size, image_size) if isinstance(image_size, int) else tuple(image_size)

    def _preprocess_frames(self, frames):
        if frames.dim() == 3:
            return self.preprocess(to_pil_image(frames))
        if frames.dim() == 4:
            return torch.stack(
                [self.preprocess(to_pil_image(frame)) for frame in frames], dim=0
            )
        raise ValueError(f"Unexpected frame shape: {tuple(frames.shape)}")

    def extract_video_features(self, video_frames):
        b, t = video_frames.shape[:2]
        frames = video_frames.cpu().reshape(b * t, *video_frames.shape[2:])
        frames = self._preprocess_frames(frames).to(self.device)
        with torch.no_grad():
            features = self.model.encode_image(frames)
        features = features.view(b, t, -1)
        features = features / features.norm(dim=-1, keepdim=True)
        video_features = torch.mean(features, dim=1)
        return video_features / video_features.norm(dim=-1, keepdim=True)

    def extract_text_features(self, texts):
        tokens = self.tokenizer(texts).to(self.device)
        text_embeddings = self.model.encode_text(tokens)
        return text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)

    def compute_similarity_matrix(self, video_features, text_embeddings):
        return torch.matmul(video_features, text_embeddings.T)
