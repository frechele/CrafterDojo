import os
import numpy as np
import pickle
import glob
import logging

from torch.utils.data import Dataset

from crafterdojo.common.utils import get_split_range
from crafterdojo.common.video_utils import load_video_to_lst
from crafterdojo.common.helpers import batch_recursive_objects
from crafterdojo.model.vpt.config import resize_image, AGENT_RESOLUTION


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

        is_valid, reason = self.is_valid()
        if not is_valid:
            raise FileNotFoundError(f"Episode {episode_id_str} is invalid: {reason}")

        with open(self.len_path, "rt") as f:
            self.len = int(f.read())

    def is_valid(self):
        if not os.path.exists(self.observation_path):
            return False, "episode video is not found"
        if not os.path.exists(self.action_path):
            return False, "episode action is not found"
        if not os.path.exists(self.len_path):
            return False, "episode len is not found"

        return True, ""

    def __len__(self) -> int:
        return self.len

    def load_frames(self, only_range=None, to_rgb: bool = True):
        return load_video_to_lst(
            self.observation_path, to_rgb, only_range, length=len(self) + 1
        )

    def load_actions(self):
        with open(self.action_path, "rb") as f:
            return pickle.load(f)


class VPTDataset(Dataset):
    def __init__(
        self,
        dataset_root: str,
        data_split: str,
        T: int,
        episode_cutoff: int = None,
        every_nth=None,
        include_partial_chunks: bool = False,
        drop_noop_actions: bool = False,
    ):
        super(VPTDataset, self).__init__()

        self.dataset_root = dataset_root
        self.drop_noop_actions = drop_noop_actions

        split_ranges = get_split_range(data_split)
        video_files = sorted(
            glob.glob(os.path.join(dataset_root, "video", "video_*.mp4"))
        )
        episode_ids = [
            int(os.path.basename(video_file).split("_")[1].split(".")[0])
            for video_file in video_files
        ]

        selected_episode_ids = []
        for start_id, end_id in split_ranges:
            selected_episode_ids.extend(episode_ids[slice(start_id, end_id)])

        self.episode_chunks = create_episode_chunks(
            dataset_root,
            selected_episode_ids,
            T,
            episode_cutoff,
            include_partial_chunks=include_partial_chunks,
        )
        if every_nth is not None:
            self.episode_chunks = self.episode_chunks[::every_nth]

        logging.info(
            f"Loaded {len(self.episode_chunks)} chunks from {len(selected_episode_ids)} episodes"
        )

        self.T = T

    def __len__(self) -> int:
        return len(self.episode_chunks)

    def __getitem__(self, idx: int) -> tuple:
        episode_id, start_idx, end_idx = self.episode_chunks[idx]
        episode = Episode(self.dataset_root, episode_id)
        valid_length = end_idx - start_idx

        frames = episode.load_frames(only_range=(start_idx, end_idx), to_rgb=True)[
            start_idx:end_idx
        ]
        obs_list = [env_obs_to_agent(frame) for frame in frames]

        action_list, is_null_list, _ = episode.load_actions()
        action_list = list(action_list[start_idx:end_idx])
        is_null_list = list(is_null_list[start_idx:end_idx])

        if valid_length <= 0:
            raise ValueError(f"episode chunk is empty: {episode_id} [{start_idx}, {end_idx})")

        if self.drop_noop_actions:
            keep = [not n for n in is_null_list]
            if any(keep):
                obs_list = [o for o, k in zip(obs_list, keep) if k]
                action_list = [a for a, k in zip(action_list, keep) if k]
                is_null_list = [False] * len(obs_list)
                valid_length = len(obs_list)

        pad_length = self.T - valid_length
        if pad_length > 0:
            last_obs = obs_list[-1]
            obs_list.extend(
                {k: v.copy() for k, v in last_obs.items()} for _ in range(pad_length)
            )
            action_list.extend([0] * pad_length)
            is_null_list.extend([False] * pad_length)

        first_list = [True] + [False] * (self.T - 1)
        valid_mask_list = [True] * valid_length + [False] * pad_length

        obs_np = batch_recursive_objects(obs_list)
        action_np = np.array(action_list, dtype=np.int64)
        is_null_np = np.array(is_null_list, dtype=bool)
        action_np[is_null_np] = 0

        first_np = np.array(first_list, dtype=bool).reshape(-1, 1)
        valid_mask_np = np.array(valid_mask_list, dtype=bool)

        for k, v in obs_np.items():
            if v.shape[0] != self.T:
                raise ValueError(f"obs shape mismatch: {episode_id} {k} {v.shape}")

        if action_np.shape[0] != self.T:
            raise ValueError(f"action shape mismatch: {episode_id} {action_np.shape}")

        if first_np.shape[0] != self.T:
            raise ValueError(f"first shape mismatch: {episode_id} {first_np.shape}")

        if is_null_np.shape[0] != self.T:
            raise ValueError(f"is_null shape mismatch: {episode_id} {is_null_np.shape}")

        if valid_mask_np.shape[0] != self.T:
            raise ValueError(
                f"valid mask shape mismatch: {episode_id} {valid_mask_np.shape}"
            )

        return obs_np, action_np, first_np, is_null_np, valid_mask_np

    def collate_fn(self, batch):
        obs_np, action_np, first_np, is_null_np, valid_mask_np = zip(*batch)
        obs = batch_recursive_objects(obs_np)
        action = batch_recursive_objects(action_np)
        first = batch_recursive_objects(first_np)
        is_null = batch_recursive_objects(is_null_np)
        valid_mask = batch_recursive_objects(valid_mask_np)

        return obs, action, first, is_null, valid_mask


def env_obs_to_agent(frame):
    agent_input = resize_image(frame, AGENT_RESOLUTION)[None]

    return {
        "img": agent_input,
    }


def create_episode_chunks(
    dataset_root: str,
    episode_ids: list[int],
    T: int,
    episode_cutoff: int = None,
    include_partial_chunks: bool = False,
):
    episode_chunks = []
    for episode_id in episode_ids:
        try:
            episode = Episode(dataset_root, episode_id)
            length = len(episode)

            if episode_cutoff is not None and length > episode_cutoff:
                length = episode_cutoff

            if include_partial_chunks:
                episode_chunks.extend(
                    (episode_id, start_idx, min(start_idx + T, length))
                    for start_idx in range(0, length, T)
                )
            else:
                # should not be overlapped
                num_chunks = length // T
                episode_chunks.extend(
                    (episode_id, i * T, (i + 1) * T) for i in range(num_chunks)
                )
        except Exception as e:
            logging.error(f"Error loading episode {episode_id}: {e}")
            continue

    return episode_chunks
