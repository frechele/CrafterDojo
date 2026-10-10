import cv2


PRIOR_INFO = {
    'mineclip_dim': 512,
    'latent_dim': 512,
    'hidden_dim': 512,
    'model_path': 'models/steve1/steve1_prior.safetensors',
}

SBERT_INFO = {
    'model_name': 'bert-base-nli-stsb-mean-tokens',
    'goal_embed_dim': 768,
}

FONT = cv2.FONT_HERSHEY_SIMPLEX
