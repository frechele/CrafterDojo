import torch
import numpy as np
import time
from collections import defaultdict
from contextlib import contextmanager

from crafterdojo.lib.VPT.lib.tree_util import tree_map


def batch_recursive_objects(ls, check_shape: bool = False):
    first = ls[0]
    if isinstance(first, dict):
        return {k: batch_recursive_objects([d[k] for d in ls]) for k in first}
    elif isinstance(first, list):
        return [
            batch_recursive_objects([lst[i] for lst in ls]) for i in range(len(first))
        ]
    elif isinstance(first, tuple):
        return tuple(
            batch_recursive_objects([lst[i] for lst in ls]) for i in range(len(first))
        )
    elif first is None:
        return None
    else:
        if check_shape:
            assert all(
                [e.shape == first.shape for e in ls]
            ), "All objects must have the same shape"

        if isinstance(first, np.ndarray):
            return np.concatenate(ls, axis=0)
        elif isinstance(first, torch.Tensor):
            return torch.cat(ls, dim=0)
        else:
            print(9)
            raise ValueError(
                f"Unsupported type: {type(first)}."
                "Only numpy arrays and torch tensors are supported "
                "for non-(dict, list, tuple, None) objects"
            )


def object_to_torch_and_device(x, device):
    def object_to_device(x):
        if isinstance(x, torch.Tensor):
            return x.to(device)
        elif isinstance(x, np.ndarray):
            return torch.from_numpy(x).to(device)

    return tree_map(object_to_device, x)


@contextmanager
def timeit_context(label):
    t0 = time.time()
    yield
    print(f"\t{label} took {time.time() - t0:.2f} seconds")


class Timer:
    """Timer class for timing code blocks.
    Stores timings based on keys passed into the context manager.
    When Timer is turned into a dict, it returns a dict of the average timings
    per key. When reset is called, it resets the timings dict."""

    def __init__(self, name):
        self.timings = defaultdict(list)
        self.throughputs_n = defaultdict(int)
        self.start_time = time.time()
        self.name = name

    @contextmanager
    def time(self, key):
        t0 = time.time()
        yield
        self.timings[key].append(time.time() - t0)

    def reset(self):
        self.timings = defaultdict(list)
        self.throughputs_n = defaultdict(int)
        self.start_time = time.time()

    def dict(self):
        cur_time = time.time()
        timings = {f"{self.name}/{k}": np.mean(v) for k, v in self.timings.items()}
        # Add the throughputs
        for k, v in self.throughputs_n.items():
            timings[f"{self.name}/{k}"] = v / (cur_time - self.start_time)
        return timings

    def time_iter(self, iterable, key):
        """Time how long it takes to get the next item from an iterable."""
        if not hasattr(iterable, "__next__"):
            iterable = iter(iterable)
        while True:
            start_time = time.time()
            try:
                item = next(iterable)
            except StopIteration:
                break
            self.timings[key].append(time.time() - start_time)
            yield item

    def throughput(self, key, n):
        """This lets us record the throughput of various operations.
        For example, the number of tokens/second processed."""
        self.throughputs_n[key] += n
