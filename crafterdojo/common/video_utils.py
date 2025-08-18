import cv2
import numpy as np


def load_video_to_lst(
    filepath: str, to_rgb: bool = False, only_range=None, length=None
):
    """Loads the frames by reading from the MP4 file.

    When only_range is specified, length cannot be None - it tells us how long to make the list,
    And, length cannot be less than only_range[1].
    """
    cap = cv2.VideoCapture(filepath)
    frames = []
    if only_range is not None:
        assert length is not None and length >= only_range[1], (
            f"length cannot be None when only_range is specified and must be "
            f">= only_range[1] ({only_range[1]}), got length={length}."
        )
        cap.set(cv2.CAP_PROP_POS_FRAMES, only_range[0])
        # Pad with nones
        for i in range(only_range[0]):
            frames.append(None)
        for i in range(only_range[0], only_range[1]):
            ret, frame = cap.read()
            if to_rgb:
                cv2.cvtColor(frame, code=cv2.COLOR_BGR2RGB, dst=frame)
            frame = np.asarray(np.clip(frame, 0, 255), dtype=np.uint8)
            if ret:
                frames.append(frame)
            else:
                break
        # Pad with nones
        for i in range(only_range[1], length):
            frames.append(None)
    else:
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                if to_rgb:
                    cv2.cvtColor(frame, code=cv2.COLOR_BGR2RGB, dst=frame)
                frame = np.asarray(np.clip(frame, 0, 255), dtype=np.uint8)
                frames.append(frame)
            else:
                break
    cap.release()
    return frames


def get_video_length(filepath: str) -> int:
    cap = cv2.VideoCapture(filepath)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return length


def save_as_mp4(frames: list, path: str, fps: int = 20, to_bgr: bool = True):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    if frames[0].shape[0] == 3:
        frames = [frame.transpose(1, 2, 0) for frame in frames]

    shape = frames[0].shape
    out = cv2.VideoWriter(path, fourcc, fps, (shape[1], shape[0]))
    for frame in frames:
        if to_bgr:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)
    out.release()
