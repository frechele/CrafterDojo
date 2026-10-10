"""VPT model config, default kwargs, and weight-file loaders."""


import cv2

from crafterdojo.common.checkpoint import load_model_config


AGENT_RESOLUTION = (144, 144)

POLICY_KWARGS = dict(
    attention_heads=16,
    attention_mask_style="clipped_causal",
    attention_memory_size=256,
    diff_mlp_embedding=False,
    hidsize=2048,
    img_shape=[144, 144, 3],
    impala_chans=[16, 32, 32],
    impala_kwargs={"post_pool_groups": 1},
    impala_width=8,
    init_norm_kwargs={"batch_norm": False, "group_norm_groups": 1},
    n_recurrence_layers=4,
    only_img_input=True,
    pointwise_ratio=4,
    pointwise_use_activation=False,
    recurrence_is_residual=True,
    recurrence_type="transformer",
    timesteps=128,
    use_pointwise_layer=True,
    use_pre_lstm_ln=False,
)

PI_HEAD_KWARGS = dict(temperature=2.0)

ACTION_TRANSFORMER_KWARGS = dict(
    camera_binsize=2,
    camera_maxval=10,
    camera_mu=10,
    camera_quantization_scheme="mu_law",
)


def resize_image(img, target_resolution):
    return cv2.resize(img, target_resolution, interpolation=cv2.INTER_LINEAR)


def load_model_parameters(path_to_model_file):
    agent_parameters = load_model_config(path_to_model_file)
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])

    lora_kwargs = None
    if "lora" in agent_parameters["model"]["args"]:
        lora_kwargs = agent_parameters["model"]["args"]["lora"]
    return policy_kwargs, pi_head_kwargs, lora_kwargs


def apply_model_overrides(policy_kwargs: dict, overrides) -> dict:
    if overrides is None:
        return policy_kwargs
    for key, value in overrides.items():
        if value is not None:
            if key not in policy_kwargs:
                raise ValueError(
                    f"model_overrides key '{key}' not found in policy_kwargs. "
                    f"Available: {sorted(policy_kwargs.keys())}"
                )
            policy_kwargs[key] = value
    return policy_kwargs
