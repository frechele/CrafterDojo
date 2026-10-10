"""CraftCLIP evaluator package (video-text retrieval).

Evaluator classes are intentionally not re-exported here to avoid pulling in heavy
optional deps (einops, transformers, open_clip) at package-import time. Import the
specific submodule you need:

    from crafterdojo.evaluator.crafterclip.clip4clip import CLIP4ClipEvaluator
    from crafterdojo.evaluator.crafterclip.mineclip import MineCLIPEvaluator
    from crafterdojo.evaluator.crafterclip.openai import OpenAICLIPEvaluator
    from crafterdojo.evaluator.crafterclip.openclip import OpenCLIPEvaluator
"""

from .retrieval import (
    compute_retrieval_metrics,
    normalize_texts_first,
    normalize_texts_join,
)
