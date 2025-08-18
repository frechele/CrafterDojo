import argparse
import glob
import json
import os
import numpy as np
import tabulate
import ray
import warnings
from collections import OrderedDict


def gather_achievements(rootpath):
    filelist = glob.glob(os.path.join(rootpath, "achievements", "*.json"))
    @ray.remote
    def _process(filename):
        with open(filename, "rt") as f:
            data = json.load(f)

        achievements = np.array([x > 0 for x in data.values()]).sum()
        diamonds = data["collect_diamond"]
        diamond_task = diamonds > 0

        return [filename, achievements, diamonds, diamond_task]

    results = [_process.remote(filename) for filename in filelist]
    results = ray.get(results)
    results = sorted(results, key=lambda x: x[0])
    results = [x[1:] for x in results]
    results = np.array(results)

    achievements_mean = results[:, 0].mean()
    achievements_std = np.std(results[:, 0])
    achievements_max = np.max(results[:, 0])

    return [len(results), achievements_mean, achievements_std, achievements_max]


def gather_achievement_success_rate(rootpath):
    filelist = glob.glob(os.path.join(rootpath, "achievements", "*.json"))
    @ray.remote
    def _process(filename):
        with open(filename, "rt") as f:
            data = json.load(f)

        achievements = { k: v > 0 for k, v in data.items() }
        return achievements

    filelist = sorted(filelist)

    full_final_result = OrderedDict()

    crafter_scores = []

    batch_size = 20
    for i in range(0, len(filelist), batch_size):
        batch_filelist = filelist[i:i+batch_size]
        results = [_process.remote(filename) for filename in batch_filelist]
        results = ray.get(results)

        final_result = OrderedDict()
        for result in results:
            for key, value in result.items():
                full_final_result[key] = full_final_result.get(key, 0) + value
                final_result[key] = final_result.get(key, 0) + value

        for key in final_result.keys():
            final_result[key] = final_result[key] / len(batch_filelist) * 100

        percents = np.fromiter(final_result.values(), dtype=np.float32)
        if (percents <= 1.0).all():
            print("Warning: All achievement success rates are less than 1.0%")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            scores = np.exp(np.nanmean(np.log(1 + percents), -1)) - 1

        crafter_scores.append(scores)

    for key in full_final_result.keys():
        full_final_result[key] = full_final_result[key] / len(filelist) * 100

    crafter_scores = np.array(crafter_scores)
    crafter_scores_mean = crafter_scores.mean()
    crafter_scores_std = np.std(crafter_scores)

    full_final_result["crafter_score"] = crafter_scores_mean
    full_final_result["crafter_score_std"] = crafter_scores_std
    
    return [full_final_result]


def gather_episode(rootpath):
    filelist = glob.glob(os.path.join(rootpath, "episode", "*.npz"))
    @ray.remote
    def _process(filename):
        data = np.load(filename)

        ep_len = len(data["reward"])
        reward_sum = data["reward"].sum()
        return [ep_len, reward_sum]

    ep_lens = []
    reward_sums = []
    max_reward = 0 

    batch_size = 20
    for i in range(0, len(filelist), batch_size):
        batch_filelist = filelist[i:i+batch_size]
        results = [_process.remote(filename) for filename in batch_filelist]
        results = np.array(ray.get(results))

        ep_lens.append(results[:, 0])

        rewards = results[:, 1] / 22
        reward_sum_mean = rewards.mean()
        reward_sums.append(rewards)

        max_reward = max(max_reward, rewards.max())

    ep_lens = np.concatenate(ep_lens)
    ep_len_mean = ep_lens.mean()
    ep_len_std = np.std(ep_lens)
    ep_len_max = ep_lens.max()

    reward_sums = np.array(reward_sums)
    reward_sum_mean = reward_sums.mean()
    reward_sum_std = np.std(reward_sums)

    return [ep_len_mean, ep_len_std, ep_len_max, reward_sum_mean, reward_sum_std, max_reward]


def clean_basename(path: str):
    if path.endswith('/'):
        path = path[:-1]
    return os.path.basename(path)


def show_statistics(args):
    results = [[clean_basename(rootpath)] + gather_achievements(rootpath) + gather_episode(rootpath)
               for rootpath in args.root]
    table = tabulate.tabulate(results, headers=["experiment", "episodes", "achievements_mean", "achievements_std", "achievements_max", "ep_len_mean", "ep_len_std", "ep_len_max", "reward_sum_mean", "reward_sum_std", "reward_sum_max"], tablefmt="grid")
    print(table)


ACHIEVEMENT_CATEGORIES = {
    "wood": ["make_wood_pickaxe", "make_wood_sword", "collect_wood", "place_table"],
    "stone": ["make_stone_pickaxe", "make_stone_sword", "collect_stone", "place_furnace", "collect_coal"],
    "iron": ["make_iron_pickaxe", "make_iron_sword", "collect_iron"],
    "diamond": ["collect_diamond"],
    "plant": ["collect_sapling", "eat_plaint", "place_plant"],
    "drink and sleep": ["collect_drink", "wake_up"],
    "combat": ["eat_cow", "defat_skeleton", "defeat_zombie"],
}
ACHIEVEMENT_MAPPING = {
    achievement: category
    for category, achievements in ACHIEVEMENT_CATEGORIES.items()
    for achievement in achievements
}
def show_achievement_per_achievement(args):
    results = [[clean_basename(rootpath)] + gather_achievement_success_rate(rootpath)
               for rootpath in args.root]

    # Individual achievements table
    headers = ["experiment"] + list(results[0][1].keys())
    results2 = []
    for experiment, archievement in results:
        numbers = []
        for key in headers[1:]:
            numbers.append(archievement.get(key, 0))
        results2.append([experiment] + numbers)

    transposed_data = list(map(list, zip(*results2)))
    transposed_headers = ["achievement"] + [r[0] for r in results2]
    table = tabulate.tabulate(transposed_data[1:], headers=transposed_headers, tablefmt="grid", showindex=headers[1:], stralign="left")
    print("Individual Achievements:")
    print(table)
    print()
    
    # Category-wise achievements table
    category_results = []
    for experiment, achievements in results:
        category_scores = {}
        
        # Calculate average success rate per category
        for category, achievement_list in ACHIEVEMENT_CATEGORIES.items():
            scores = []
            for achievement in achievement_list:
                if achievement in achievements:
                    scores.append(achievements[achievement])
            
            if scores:  # Only if we have data for this category
                category_scores[category] = np.mean(scores)
            else:
                category_scores[category] = 0
        
        category_results.append([experiment, category_scores])
    
    # Build category table
    category_headers = ["experiment"] + list(ACHIEVEMENT_CATEGORIES.keys())
    category_results2 = []
    for experiment, category_scores in category_results:
        numbers = []
        for category in ACHIEVEMENT_CATEGORIES.keys():
            numbers.append(category_scores.get(category, 0))
        category_results2.append([experiment] + numbers)
    
    category_transposed = list(map(list, zip(*category_results2)))
    category_transposed_headers = ["category"] + [r[0] for r in category_results2]
    category_table = tabulate.tabulate(category_transposed[1:], headers=category_transposed_headers, tablefmt="grid", showindex=list(ACHIEVEMENT_CATEGORIES.keys()), stralign="left")
    print("Achievement Categories (Average Success Rate):")
    print(category_table)




def main(args):
    ray.init()

    print("[Statistics]")
    show_statistics(args)
    print()

    print("[Achievement]")
    show_achievement_per_achievement(args)
    print()

    ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", metavar="path", type=str, nargs="+")
    args = parser.parse_args()
    main(args)
