import os
import pickle
import glob
import logging
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

from crafterdojo.common.utils import get_split_range
from crafterdojo.common.video_utils import load_video_to_lst
from crafterdojo.common.helpers import batch_recursive_objects


class Episode:
    def __init__(self, dataset_root: str, caption_path: str, episode_id: int):
        self.dataset_root = dataset_root

        episode_id_str = f"{episode_id:05d}"
        self.video_path = os.path.join(
            dataset_root, "video", f"video_{episode_id_str}.mp4"
        )
        self.caption_path = os.path.join(
            dataset_root, caption_path, f"episode_{episode_id_str}.pkl"
        )
        self.len_path = os.path.join(
            dataset_root, "len", f"len_{episode_id_str}.txt"
        )

        assert self.is_valid()

        with open(self.len_path, "rt") as f:
            self.len = int(f.read())

    def is_valid(self):
        if not os.path.exists(self.video_path):
            return False, "video file does not exist"
        if not os.path.exists(self.caption_path):
            return False, "caption file does not exist"
        if not os.path.exists(self.len_path):
            return False, "len file does not exist"

        return True, ""

    def __len__(self) -> int:
        return self.len

    def load_frames(self, only_range=None, to_rgb: bool = True):
        return load_video_to_lst(self.video_path, to_rgb, only_range, length=len(self))

    def load_captions(self):
        with open(self.caption_path, "rb") as f:
            captions = pickle.load(f)
        return captions

class ClipCaptionDataset(Dataset):
    def __init__(
        self,
        dataset_root: str,
        caption_path: str,
        mode: str,
        data_split: str,
        resize_resolution: tuple[int, int],
        episode_cutoff: int = None,
        every_nth=None,
        augmentation: bool = False,
    ):
        super(ClipCaptionDataset, self).__init__()

        self.dataset_root = dataset_root
        self.resize_resolution = resize_resolution
        self.mode = mode  # choices=["train", "val", "test"]
        self.augmentation = augmentation
        self.caption_path = caption_path

        start_id, end_id = get_split_range(data_split)

        caption_files = sorted(
            glob.glob(os.path.join(dataset_root, caption_path, "episode_*.pkl"))
        )
        episode_ids = [
            int(os.path.basename(caption_file).split("_")[1].split(".")[0])
            for caption_file in caption_files
        ]

        episode_ids = episode_ids[start_id:end_id]
        self.episode_chunks = create_episode_chunks(
            dataset_root, caption_path, episode_ids, episode_cutoff
        )
        if every_nth is not None:
            self.episode_chunks = self.episode_chunks[::every_nth]

        logging.info(
            f"Loaded {len(self.episode_chunks)} chunks from {len(episode_ids)} episodes ({self.mode} set)"
        )

    def __len__(self) -> int:
        return len(self.episode_chunks)

    def __getitem__(self, idx: int) -> tuple:
        episode_id, start_frame, end_frame, description, metadata = self.episode_chunks[idx]

        if not isinstance(metadata, dict):
            metadata = { "rule_type": metadata }
        
        episode = Episode(self.dataset_root, self.caption_path, episode_id)

        frames = episode.load_frames(only_range=(start_frame, end_frame), to_rgb=True)[
            start_frame:end_frame
        ]

        crop_params = None
        obs_list = [
            env_obs_to_agent(
                frame, self.resize_resolution, self.augmentation, crop_params
            )
            for frame in frames
        ]

        obs_np = batch_recursive_objects(obs_list)

        metadata.update({
            "episode_id": episode_id,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "target_text": description,
        })

        return obs_np, description, metadata

    def collate_fn(self, batch):
        obs, caption, metadata = zip(*batch)
        obs = torch.stack([torch.as_tensor(o) for o in obs])
        caption = list(caption)
        metadata = list(metadata)
        return obs, caption, metadata


def get_crop_params(frame, crop_size, resize_min=272):
    """
    Get parameters for temporally-consistent random resized crop
    """
    H, W = frame.shape[:2]
    if W < H:
        new_W = resize_min
        new_H = int(H * (new_W / W))
    else:
        new_H = resize_min
        new_W = int(W * (new_H / H))

    assert (crop_size[0] <= new_H) and (
        crop_size[1] <= new_W
    ), "Crop size is larger than the frame size"

    base_y = (new_H - crop_size[0]) // 2
    base_x = (new_W - crop_size[1]) // 2

    max_shift_x_left = base_x
    max_shift_x_right = new_W - crop_size[1] - base_x
    max_shift_y_top = base_y
    max_shift_y_bottom = new_H - crop_size[0] - base_y

    shift_x = np.random.randint(-max_shift_x_left, max_shift_x_right + 1)
    shift_y = np.random.randint(-max_shift_y_top, max_shift_y_bottom + 1)

    crop_x = base_x + shift_x
    crop_y = base_y + shift_y

    return (new_H, new_W), (crop_x, crop_y), crop_size


def env_obs_to_agent(
    frame,
    resize_resolution: tuple[int, int],
    augmentation: bool,
    crop_params: tuple[tuple[int, int], tuple[int, int], tuple[int, int]] = None,
):
    if (
        augmentation and crop_params is not None
    ):  # Apply temporally-consistent random resized crop
        (new_H, new_W), (crop_x, crop_y), (crop_H, crop_W) = crop_params
        tmp_frame = resize_image(frame, (new_H, new_W))
        resized_frame = tmp_frame[crop_y : crop_y + crop_H, crop_x : crop_x + crop_W][
            None
        ]
    else:
        resized_frame = resize_image(frame, resize_resolution)[None]

    # frame is initially B x H x W x C, convert to B x C x H x W.
    resized_frame = resized_frame.transpose(0, 3, 1, 2)
    return resized_frame


def create_episode_chunks(
    dataset_root: str, caption_path: str, episode_ids: list[int], episode_cutoff: int = None
):
    episode_chunks = []
    for episode_id in episode_ids:
        try:
            episode = Episode(dataset_root, caption_path, episode_id)
            length = len(episode)

            if episode_cutoff is not None and length > episode_cutoff:
                length = episode_cutoff

            captions = episode.load_captions()

            for caption in captions:
                (start_frame, end_frame), description, metadata = caption
                episode_chunks.append((episode_id, start_frame, end_frame, description, metadata))

        except Exception as e:
            logging.error(f"Error loading episode {episode_id}: {e}")
            continue

    return episode_chunks


def resize_image(img, target_resolution):
    # For your sanity, do not resize with any function than INTER_LINEAR
    img = cv2.resize(img, target_resolution, interpolation=cv2.INTER_LINEAR)
    return img