import numpy as np
import torch as th
import cv2
import pickle
from gym3.types import discrete_scalar

from crafterdojo.lib.VPT.lib.policy import MinecraftAgentPolicy
from crafterdojo.lib.VPT.lib.torch_util import default_device_type, set_default_torch_device


# Hardcoded settings
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
    # For your sanity, do not resize with any function than INTER_LINEAR
    img = cv2.resize(img, target_resolution, interpolation=cv2.INTER_LINEAR)
    return img


def load_model_parameters(path_to_model_file):
    agent_parameters = pickle.load(open(path_to_model_file, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])

    lora_kwargs = None
    if "lora" in agent_parameters["model"]["args"]:
        lora_kwargs = agent_parameters["model"]["args"]["lora"]
    return policy_kwargs, pi_head_kwargs, lora_kwargs


class CrafterRLAgent:
    def __init__(self, device=None, policy_kwargs=None, pi_head_kwargs=None, lora_kwargs=None):
        if device is None:
            device = default_device_type()
        self.device = th.device(device)
        set_default_torch_device(self.device)
        action_space = discrete_scalar(17)

        if policy_kwargs is None:
            policy_kwargs = POLICY_KWARGS
        if pi_head_kwargs is None:
            pi_head_kwargs = PI_HEAD_KWARGS

        agent_kwargs = dict(policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs, action_space=action_space)

        self.policy = MinecraftAgentPolicy(**agent_kwargs).to(device)

        if lora_kwargs is not None:
            self.policy.inject_lora(lora_alpha=lora_kwargs["alpha"], lora_r=lora_kwargs["r"])

        self.hidden_state = self.policy.initial_state(1)
        self._dummy_first = th.from_numpy(np.array((False,))).to(device)

    def load_weights(self, path):
        """Load model weights from a path, and reset hidden state"""
        self.policy.load_state_dict(th.load(path, map_location=self.device, weights_only=True), strict=False)
        self.reset()

    def reset(self):
        """Reset agent to initial state (i.e., reset hidden state)"""
        self.hidden_state = self.policy.initial_state(1)

    def _env_obs_to_agent(self, minerl_obs):
        """
        Turn observation from MineRL environment into model's observation

        Returns torch tensors.
        """
        agent_input = minerl_obs["pov"][None]
        agent_input = {"img": th.from_numpy(agent_input).to(self.device)}
        return agent_input

    def get_action(self, minerl_obs, return_result: bool = False):
        """
        Get agent's action for given MineRL observation.

        Agent's hidden state is tracked internally. To reset it,
        call `reset()`.
        """
        agent_input = self._env_obs_to_agent(minerl_obs)

        agent_action, self.hidden_state, result = self.policy.act(
            agent_input, self._dummy_first, self.hidden_state,
            stochastic=True, return_pd=True
        )
        if return_result:
            return agent_action, result
        return agent_action
