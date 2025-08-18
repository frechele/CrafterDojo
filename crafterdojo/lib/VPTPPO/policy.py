from typing import Callable, Dict, List, Optional, Tuple, Type, Union
from gym3.types import TensorType, Discrete
import gymnasium as gym
import torch as th
from torch import nn
from itertools import chain

from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy

from .types import VPTStates
from crafterdojo.lib.VPT.lib.policy import MinecraftAgentPolicy
from crafterdojo.lib.VPT.lib.tree_util import tree_map


class VPTPolicy(RecurrentActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        config,
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        *args,
        **kwargs
    ):
        policy_kwargs = kwargs.pop("policy_kwargs", dict())
        pi_head_kwargs = kwargs.pop("pi_head_kwargs", dict())
        weights_path = kwargs.pop("weights_path", None)
        vpt_action_space = TensorType(shape=(1,), eltype=Discrete(action_space.n))
        lora_kwargs = kwargs.pop("lora_kwargs", None)

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            *args,
            **kwargs,
        )
        
        self.model = MinecraftAgentPolicy(
            policy_kwargs=policy_kwargs, 
            pi_head_kwargs=pi_head_kwargs, 
            action_space=vpt_action_space
        )
        self.exploration_model = MinecraftAgentPolicy(
            policy_kwargs=policy_kwargs, 
            pi_head_kwargs=pi_head_kwargs, 
            action_space=vpt_action_space
        )
        if weights_path:
            self.model.load_state_dict(th.load(weights_path), strict=False)
            self.exploration_model.load_state_dict(th.load(weights_path), strict=False)

        self.model.requires_grad_(False)
        self.exploration_model.requires_grad_(False)
        self.params = {}

        self.model.value_head.reset_parameters()

        if lora_kwargs is not None:
            self.model.inject_lora(lora_kwargs["alpha"], lora_kwargs["r"])

            self.model.pi_head.requires_grad_(True)
            self.params["model.pi_head"] = self.model.pi_head.parameters()

            self.model.value_head.requires_grad_(True)
            self.params["model.value_head"] = self.model.value_head.parameters()

            for n, x in self.model.net.recurrent_layer.named_modules():
                if any(p.requires_grad for p in x.parameters(recurse=False)):
                    self.params["model.net.recurrent_layer." + n] = x.parameters()
        else:
            self.model.requires_grad_(True)
            self.params["model"] = self.model.parameters()

        self.optimizer = self.optimizer_class(
            chain(*self.params.values()), 
            lr=lr_schedule(1), 
            **self.optimizer_kwargs
        )

        # count params
        num_param_grad = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        num_param = sum(p.numel() for p in self.model.parameters())
        print("Model setup. Params: {}. Optimize params: {} ({:.2f}%).".format(num_param, num_param_grad, num_param_grad / num_param * 100))

    def get_param_keys(self) -> List[str]:
        return list(self.params.keys())

    @staticmethod
    def _vpt_states_to_sb3(states):
        st = ([], [], [])
        for block_st in states:
            if block_st[0] is None:
                st[0].append(th.full_like(block_st[1][0], -1)[:, :, 0])
            else:
                assert block_st[0].shape[1] == 1
                st[0].append(block_st[0][:, 0])
            st[1].append(block_st[1][0])
            st[2].append(block_st[1][1])
        st = tuple([
            th.cat([blk.unsqueeze(0) for blk in state], dim=0)
            for state in st
        ])
        return VPTStates(*st)

    @staticmethod
    def _sb3_states_to_vpt(states):
        return tuple([
            (
                None if th.all(states[0][i] == -1) else \
                    states[0][i].unsqueeze(1).bool() if len(states[0][i].shape) == 2 else \
                    states[0][i].bool(), 
                (states[1][i], states[2][i])
            )
            for i in range(states[0].shape[0])
        ])

    def initial_state(self, batch_size):
        return self._vpt_states_to_sb3(self.model.initial_state(batch_size))

    def forward(self, 
        obs: th.Tensor,             # batch x H x W x C
        in_states: VPTStates,       # n_blocks x 1 x buffer, n_blocks x 1 x buffer x hidden
        episode_starts: th.Tensor,  # batch
        deterministic: bool = False
    ) -> Tuple[Dict[str, th.Tensor], th.Tensor, th.Tensor, VPTStates, Dict[str, th.Tensor]]:
        # pd: dict: batch x 1 x 1 x 121, batch x 1 x 1 x 8641
        # vpred: batch x 1 x 1
        (pd, vpred, _), state_out = self.model(
            tree_map(lambda x: x.unsqueeze(1), obs),
            episode_starts.unsqueeze(-1).bool(),
            self._sb3_states_to_vpt(in_states)
        )

        ac = self.model.pi_head.sample(pd, deterministic=deterministic) # dict: batch x 1 x 1
        log_prob = self.model.pi_head.logprob(ac, pd)[:, 0]             # batch
        vpred = vpred[:, 0, 0]                                          # batch
        #print(ac, log_prob, vpred)
        # ac = th.cat([x[:, 0] for x in ac.to_sparse().values()], dim=1)              # batch x 2
        ac = ac[..., 0, 0]
        #print(ac)
        return ac, vpred, log_prob, self._vpt_states_to_sb3(state_out)

    def predict_values(
        self,
        obs: th.Tensor,
        in_states: VPTStates,
        episode_starts: th.Tensor,
    ) -> th.Tensor:
        (_, vpred, _), _ = self.model(
            tree_map(lambda x: x.unsqueeze(1), obs),
            episode_starts.unsqueeze(-1).bool(),
            self._sb3_states_to_vpt(in_states),
        )
        return vpred[:, 0, 0]  # batch x 1
        
    def evaluate_actions(
        self,
        obs: th.Tensor,             # n_seq * max_len x H x W x C
        actions: th.Tensor,         # n_seq * max_len x 2
        in_states: VPTStates,       # n_blocks x n_seq x buffer, n_blocks x n_seq x buffer x hidden
        episode_starts: th.Tensor,  # n_seq * max_len
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:

        n_seq = in_states[0].shape[1]
        obs_sequence = { k: v.reshape((n_seq, -1) + v.shape[1:])
                         for k, v in obs.items() }
        max_len = obs_sequence["img"].shape[1]
        starts_sequence = episode_starts.reshape((n_seq, max_len))  # n_seq x max_len
        model_input = obs_sequence, starts_sequence.bool(), self._sb3_states_to_vpt(in_states)

        # pd: dict: n_seq x max_len x 1 x 121, n_seq x max_len x 1 x 8641
        # vpred: n_seq x max_len x 1
        (pd, vpred, _), _ = self.model(*model_input)

        with th.no_grad():
            (exploration_pd, _, _), _ = self.exploration_model(*model_input)
        
        actions_reshaped = actions.reshape(n_seq, max_len, 1)
        
        log_prob = self.model.pi_head.logprob(actions_reshaped, pd)     # n_seq x max_len
        kl = self.model.get_kl_of_action_dists(pd, exploration_pd)

        return th.flatten(vpred), th.flatten(log_prob), th.flatten(kl)

    def _predict(
        self,
        observation: th.Tensor,
        vpt_states: VPTStates,
        episode_starts: th.Tensor,
        deterministic: bool = False,
    ) -> Tuple[th.Tensor, VPTStates]:
        """
        Get the action according to the policy for a given observation.

        :param observation:
        :param vpt_states: The VPT states for the model.
        :param episode_starts: Whether the observations correspond to new episodes
            or not (we reset the states in that case).
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy and VPT states
        """
        actions, values, log_prob, new_states = self.forward(
            observation, vpt_states, episode_starts, deterministic=deterministic
        )
        return actions, new_states

    def predict(
        self,
        observation: Union[th.Tensor, dict],
        state: Optional[VPTStates] = None,
        episode_start: Optional[th.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[th.Tensor, VPTStates]:
        """
        Get the policy action from an observation (and optional VPT state).
        Includes sugar-coating to handle different observations.

        :param observation: the input observation
        :param state: The VPT states for the model.
        :param episode_start: Whether the observations correspond to new episodes
            or not (we reset the states in that case).
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next VPT state
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.set_training_mode(False)

        if not isinstance(observation, th.Tensor):
            # Convert observation to tensor if needed
            if isinstance(observation, dict):
                observation = {k: th.tensor(v, dtype=th.float32, device=self.device) 
                             for k, v in observation.items()}
            else:
                observation = th.tensor(observation, dtype=th.float32, device=self.device)

        # Get batch size
        if isinstance(observation, dict):
            batch_size = observation[next(iter(observation.keys()))].shape[0]
        else:
            batch_size = observation.shape[0]

        # Initialize states if None
        if state is None:
            state = self.initial_state(batch_size)

        # Initialize episode_start if None
        if episode_start is None:
            episode_start = th.zeros(batch_size, dtype=th.float32, device=self.device)
        elif not isinstance(episode_start, th.Tensor):
            episode_start = th.tensor(episode_start, dtype=th.float32, device=self.device)

        with th.no_grad():
            actions, new_states = self._predict(
                observation, vpt_states=state, episode_starts=episode_start, deterministic=deterministic
            )

        return actions, new_states
