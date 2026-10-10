import torch as th
from torch import nn
from torch.nn import functional as F

from crafterdojo.model.vpt.lib.policy import (
    InverseActionNet as VPTInverseActionNet,
    InverseActionPolicy as VPTInverseActionPolicy,
    MinecraftAgentPolicy as VPTMinecraftAgentPolicy,
    MinecraftPolicy as VPTMinecraftPolicy,
)
from crafterdojo.model.vpt.lib.tree_util import tree_map


class GoalConditionedMinecraftPolicy(VPTMinecraftPolicy):
    """Goal-conditioned VPT policy.

    cond_position selects where the projected goal embedding is injected:
      - "output": added to the backbone output latent (after final_ln). Default.
      - "input":  added to img features before pre_lstm_ln / recurrent layer,
                  matching upstream Steve-1 (Shalev-Lifshitz/STEVE-1).
    """

    def __init__(
        self,
        goal_embed_dim: int = 512,
        cond_position: str = "output",
        **kwargs,
    ):
        super().__init__(**kwargs)
        if cond_position not in ("output", "input"):
            raise ValueError(
                f"cond_position must be 'output' or 'input', got {cond_position!r}"
            )
        self.cond_position = cond_position
        self.mineclip_embed_linear = nn.Linear(goal_embed_dim, self.hidsize)

    def _project_goal(self, ob, b, t):
        goal_embed = ob["goal_embed"].reshape(b * t, -1)
        goal_embed = self.mineclip_embed_linear(goal_embed)
        return goal_embed.reshape(b, t, -1)

    def forward(self, ob, state_in, context):
        b, t = ob["img"].shape[:2]
        first = context["first"].bool()

        if self.cond_position == "output":
            goal_embed = self._project_goal(ob, b, t)
            output, state_out = super().forward(
                ob, state_in, context={"first": first}
            )
            if self.single_output:
                return output + goal_embed, state_out
            pi_latent, vf_latent = output
            return (pi_latent + goal_embed, vf_latent + goal_embed), state_out

        # cond_position == "input": replicate parent forward, inject goal after
        # img_process and before pre_lstm_ln / recurrent_layer.
        x = self.img_preprocess(ob["img"])
        x = self.img_process(x)

        if self.diff_obs_process is not None:
            x = self.diff_obs_process(ob["diff_goal"]) + x

        x = x + self._project_goal(ob, b, t)

        if self.pre_lstm_ln is not None:
            x = self.pre_lstm_ln(x)

        if self.recurrent_layer is not None:
            x, state_out = self.recurrent_layer(x, first, state_in)
        else:
            state_out = state_in

        x = F.relu(x, inplace=False)
        x = self.lastlayer(x)
        x = self.final_ln(x)

        if self.single_output:
            return x, state_out
        return (x, x), state_out


class MinecraftAgentPolicy(VPTMinecraftAgentPolicy):
    def __init__(self, action_space, policy_kwargs, pi_head_kwargs):
        super().__init__(
            action_space=action_space,
            policy_kwargs=policy_kwargs,
            pi_head_kwargs=pi_head_kwargs,
            policy_cls=GoalConditionedMinecraftPolicy,
        )
        self.debug = False

    @property
    def goal_projection(self) -> nn.Module:
        return self.net.mineclip_embed_linear

    def get_output_for_observation(self, obs, state_in, first):
        (pd, vpred, _), state_out = self(obs=obs, first=first, state_in=state_in)
        return pd, self.value_head.denormalize(vpred)[:, 0], state_out

    @th.no_grad()
    def act(
        self,
        obs,
        first,
        state_in,
        stochastic: bool = True,
        taken_action=None,
        return_pd: bool = False,
        cond_scale=None,
    ):
        if cond_scale is None or cond_scale == 0:
            return super().act(
                obs,
                first,
                state_in,
                stochastic=stochastic,
                taken_action=taken_action,
                return_pd=return_pd,
            )

        bsz = obs["img"].shape[0]
        assert bsz == 1, "cond_scale only works for batch size 1"

        guided_obs = tree_map(lambda x: th.cat([x, x], dim=0), obs)
        guided_obs["goal_embed"][1] = th.zeros_like(guided_obs["goal_embed"][1])
        guided_first = th.cat([first, first], dim=0)

        guided_obs = tree_map(lambda x: x.unsqueeze(1), guided_obs)
        guided_first = guided_first.unsqueeze(1)

        (pd, vpred, _), state_out = self(obs=guided_obs, first=guided_first, state_in=state_in)
        pd = tree_map(lambda x: self._apply_classifier_free_guidance(x[0], x[1], cond_scale), pd)

        if taken_action is None:
            ac = self.pi_head.sample(pd, deterministic=not stochastic)
        else:
            ac = tree_map(lambda x: x.unsqueeze(1), taken_action)

        log_prob = self.pi_head.logprob(ac, pd)
        result = {"log_prob": log_prob[:, 0], "vpred": self.value_head.denormalize(vpred)[:, 0]}

        if return_pd:
            result["pd"] = tree_map(lambda x: x[:, 0], pd)
        ac = tree_map(lambda x: x[:, 0], ac)

        return ac, state_out, result

    @staticmethod
    def _apply_classifier_free_guidance(conditional: th.Tensor, unconditional: th.Tensor, cond_scale: float):
        guided = ((1 + cond_scale) * conditional) - (cond_scale * unconditional)
        guided = guided.unsqueeze(0)
        guided = guided - th.logsumexp(guided, dim=-1, keepdim=True)
        return guided


InverseActionNet = VPTInverseActionNet
InverseActionPolicy = VPTInverseActionPolicy
