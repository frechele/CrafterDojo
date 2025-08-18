import argparse
import glob
import os
import numpy as np
import tabulate
import ray
import pickle
from omegaconf import OmegaConf
from scipy.stats import sem


STEVE1_MODE = 0


def gather_statistics(rootpath):
    if STEVE1_MODE:
        config_path = os.path.join(rootpath, ".hydra", "config.yaml")
        if os.path.exists(config_path):
            config = OmegaConf.load(config_path)

            weights = config.agent.weights
            inst = config.agent.eval.instruction
            cond_scale = config.agent.eval.cond_scale
        else:
            weights = "unk"
            inst = "unk"
            cond_scale = "unk"

    filelist = glob.glob(os.path.join(rootpath, "info", "*.pkl"))
    @ray.remote
    def _process(filename):
        with open(filename, "rb") as f:
            data = pickle.load(f)

        ep_len = len(data)
        if "is_success" in data[-1]:
            success = data[-1]["is_success"]
        else:
            success = data[-1]["success"]
        return [ep_len, success]

    results = [_process.remote(filename) for filename in filelist]
    results = ray.get(results)
    results = np.array(results)

    if len(results) == 0:
        ep_len_mean = float("nan")
        ep_len_sem = float("nan")
        ep_len_max = float("nan")

        success_ep_lens = float("nan")
        success_ep_lens_mean = float("nan")

        success_ep_lens_sem = float("nan")
        success_ep_lens_max = float("nan")

        success_mean = float("nan")
    else:
        ep_len_mean = results[:, 0].mean()
        ep_len_sem = sem(results[:, 0])
        ep_len_max = np.max(results[:, 0])

        success_ep_lens = results[:, 0][results[:, 1] == 1]
        success_ep_lens_mean = success_ep_lens.mean()
        if len(success_ep_lens) == 0:
            success_ep_lens_sem = float("nan")
            success_ep_lens_max = float("nan")
        else:
            success_ep_lens_sem = sem(success_ep_lens)
            success_ep_lens_max = np.max(success_ep_lens)

        success_mean = results[:, 1].mean()

    if STEVE1_MODE:
        return [os.path.basename(weights), inst, cond_scale, len(filelist), ep_len_mean, ep_len_sem, ep_len_max, success_ep_lens_mean, success_ep_lens_sem, success_ep_lens_max, success_mean]
    return [len(filelist), ep_len_mean, ep_len_sem, ep_len_max, success_ep_lens_mean, success_ep_lens_sem, success_ep_lens_max, success_mean]


def clean_basename(path: str):
    if path.endswith('/'):
        path = path[:-1]
    return os.path.basename(path)


def main(args):
    results = [[clean_basename(rootpath)] + gather_statistics(rootpath)
               for rootpath in args.root]
    
    if STEVE1_MODE:
        headers = ["experiment", "weights", "instruction", "cond_scale", "episodes", "ep_len_mean", "ep_len_sem", "ep_len_max", 
                   "success_ep_lens_mean", "success_ep_lens_sem", "success_ep_lens_max", 
                   "success_rate"]
    else:
        headers = ["experiment", "episodes", "ep_len_mean", "ep_len_sem", "ep_len_max", 
                   "success_ep_lens_mean", "success_ep_lens_sem", "success_ep_lens_max", 
                   "success_rate"]
    
    experiment_names = [row[0] for row in results]
    data_matrix = [row[1:] for row in results]
    
    # Transpose the data
    transposed_data = list(map(list, zip(*data_matrix)))
    
    transposed_results = []
    metric_names = headers[1:]
    
    for i, metric_name in enumerate(metric_names):
        row = [metric_name] + transposed_data[i]
        transposed_results.append(row)
    
    transposed_headers = ["metric"] + experiment_names
    table = tabulate.tabulate(transposed_results, headers=transposed_headers, tablefmt="grid")
    
    print(table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=str, nargs="+")
    args = parser.parse_args()
    main(args)
