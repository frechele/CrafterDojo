import os
import numpy as np
import pickle 
import glob
import logging

from torch.utils.data import Dataset

from crafterdojo.common.utils import get_split_range
from crafterdojo.common.video_utils import get_video_length, load_video_to_lst
from crafterdojo.common.helpers import batch_recursive_objects
from crafterdojo.lib.VPT.agent import resize_image, AGENT_RESOLUTION


NONE_EMBED_OFFSET = 5


class Episode:
    def __init__(self, dataset_root: str, episode_id: int):
        self.dataset_root = dataset_root

        episode_id_str = f"{episode_id:05d}"
        self.observation_path = os.path.join(
            self.dataset_root, "video", f"video_{episode_id_str}.mp4"
        )
        self.action_path = os.path.join(
            self.dataset_root, "action", f"action_{episode_id_str}.pkl"
        )
        self.len_path = os.path.join(
            self.dataset_root, "len", f"len_{episode_id_str}.txt"
        )
        self.embedding_path = os.path.join(
            self.dataset_root, f"clip_embedding", f"{episode_id_str}.npz"
        )
        self.event_path = os.path.join(
            self.dataset_root, "event", f"event_{episode_id_str}.pkl"
        )

        assert self.is_valid()

        with open(self.len_path, "rt") as f:
            self.len = int(f.read())

    def is_valid(self):
        if not os.path.exists(self.observation_path):
            return False, "episode video is not found"
        if not os.path.exists(self.action_path):
            return False, "episode action is not found"
        if not os.path.exists(self.len_path):
            return False, "episode len is not found"
        if not os.path.exists(self.embedding_path):
            return False, "episode embedding is not found"
        if not os.path.exists(self.event_path):
            return False, "episode event is not found"

        return True, ""

    def __len__(self) -> int:
        return self.len

    def load_frames(self, only_range=None, to_rgb: bool = True):
        return load_video_to_lst(
            self.observation_path, to_rgb, only_range, length=len(self)
        )

    def load_actions(self):
        with open(self.action_path, "rb") as f:
            return pickle.load(f)

    def load_embedding(self):
        return np.load(self.embedding_path)["embeddings"]

    def load_events(self):
        with open(self.event_path, "rb") as f:
            return pickle.load(f)


class Steve1Dataset(Dataset):
    def __init__(
        self,
        dataset_root: str,
        sampling_path: str,
        T: int,
        min_btwn_goals: int,
        max_btwn_goals: int,
        p_uncond: float,
        event_based_goals: bool,
        episode_cutoff: int = None,
        every_nth: int = None,
    ):
        assert min_btwn_goals <= max_btwn_goals, "minimum number of timesteps between goals must be less than or equal to maximum"

        self.dataset_root = dataset_root
        self.T = T
        self.min_btwn_goals = min_btwn_goals
        self.max_btwn_goals = max_btwn_goals
        self.p_uncond = p_uncond
        self.event_based_goals = event_based_goals

        split_range = get_split_range(sampling_path)
        self.episode_chunks = create_episode_chunks(
            dataset_root, split_range, T, min_btwn_goals, episode_cutoff
        )
        if every_nth is not None:
            self.episode_chunks = self.episode_chunks[::every_nth]

        logging.info(
            f"Created {len(self.episode_chunks) / 1e3:.2f}K episode chunks from {sampling_path}."
        )

    def __len__(self) -> int:
        return len(self.episode_chunks)
    
    def __getitem__(self, idx: int):
        obs_np, actions_np, firsts_np = get_episode_chunk(
            self.dataset_root,
            self.episode_chunks[idx],
            self.T,
            self.min_btwn_goals,
            self.max_btwn_goals,
            self.p_uncond,
            self.event_based_goals,
        )

        return obs_np, actions_np, firsts_np

    def collate_fn(self, batch):
        obs_np, actions_np, firsts_np = zip(*batch)
        obs = batch_recursive_objects(obs_np)
        actions = batch_recursive_objects(actions_np)
        firsts = batch_recursive_objects(firsts_np)

        return obs, actions, firsts


def env_obs_to_agent(frame, goal_embed):
    agent_input = resize_image(frame, AGENT_RESOLUTION)[None]

    return {
        "img": agent_input,
        "goal_embed": goal_embed,
    }


def get_episode_chunk(
    dataset_root: str,
    episode_chunk,
    T: int,
    min_btwn_goals: int,
    max_btwn_goals: int,
    p_uncond: float,
    event_based_goals: bool = False,
):
    episode_id, start_timestep, end_timestep = episode_chunk
    chunk_length = end_timestep - start_timestep
    episode = Episode(dataset_root, episode_id)

    embeds = episode.load_embedding()
    frames = episode.load_frames(only_range=(start_timestep, end_timestep))
    total_timesteps = len(episode)

    # Choose goal timesteps
    if event_based_goals:
        events = [0] + episode.load_events() + [total_timesteps]
        goal_timesteps = []
        for evt_start, evt_end in zip(events[:-1], events[1:]):
            cur_timestep = evt_start

            while cur_timestep < evt_end:
                cur_timestep += np.random.randint(min_btwn_goals, max_btwn_goals) if min_btwn_goals < max_btwn_goals else 1
                if cur_timestep > evt_end:
                    cur_timestep = evt_end

                goal_timesteps.append(cur_timestep)
    else:
        goal_timesteps = []
        cur_timestep = 0
        while cur_timestep < total_timesteps - 1:
            cur_timestep += np.random.randint(min_btwn_goals, max_btwn_goals) if min_btwn_goals < max_btwn_goals else 1

            if (total_timesteps - cur_timestep) < min_btwn_goals:
                cur_timestep = total_timesteps - 1
            goal_timesteps.append(cur_timestep)

    embeds_per_timestep = []
    cur_goal_timestep_idx = 0
    for t in range(total_timesteps):
        if cur_goal_timestep_idx < len(goal_timesteps):
            goal_timestep = goal_timesteps[cur_goal_timestep_idx]
        else:
            goal_timestep = -1

        embed = embeds[goal_timestep]

        embeds_per_timestep.append(embed)
        if t == goal_timestep - 1:
            cur_goal_timestep_idx += 1

    # With probability p_uncond, set the embeds to zero
    if np.random.rand() < p_uncond:
        embeds_per_timestep = [np.zeros_like(embed) for embed in embeds_per_timestep]

    # Load the actions
    all_actions, is_null_list, _ = episode.load_actions()

    obs_list = []
    actions_list = []
    is_nulls_list = []
    firsts_list = [True] + [False] * (chunk_length - 1)

    # Only iterate over the needed range
    for i, t in enumerate(range(start_timestep, end_timestep)):
        frame = frames[t]

        obs = env_obs_to_agent(frame, embeds_per_timestep[t].reshape(1, -1))
        obs_list.append(obs)

        action = all_actions[t]
        actions_list.append(action)

        is_null = is_null_list[t]
        is_nulls_list.append(is_null)

    obs_np = batch_recursive_objects(obs_list)
    actions_np = np.array(actions_list, dtype=np.int64)
    firsts_np = np.array(firsts_list, dtype=bool).reshape(chunk_length, 1)
    is_null_np = np.array(is_nulls_list, dtype=bool)
    actions_np[is_null_np] = 0

    return obs_np, actions_np, firsts_np


def create_episode_chunks(
    dataset_root: str,
    split_range: tuple[int, int],
    T: int,
    min_btwn_goals: int,
    episode_cutoff: int = None,
):
    min_len = max(min_btwn_goals + NONE_EMBED_OFFSET, T + NONE_EMBED_OFFSET)

    episode_chunks = []
    start_id, end_id = split_range
    for episode_id in range(start_id, end_id):
        episode = Episode(dataset_root, episode_id)
        if not episode.is_valid():
            continue

        episode_length = len(episode)
        if episode_length < min_len:
            continue

        if episode_cutoff is not None and episode_length > episode_cutoff:
            episode_length = episode_cutoff

        for chunk_idx in range(NONE_EMBED_OFFSET, episode_length, T):
            start_timestep = chunk_idx
            end_timestep = start_timestep + T
            if end_timestep >= episode_length:
                start_timestep = episode_length - T
                end_timestep = episode_length
            episode_chunks.append((episode_id, start_timestep, end_timestep))
        
    return episode_chunks
