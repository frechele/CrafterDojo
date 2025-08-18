import torch


def compute_metrics(x: torch.Tensor, labels: torch.Tensor):
    N = x.shape[0]

    sorted_indices = torch.argsort(x, dim=1, descending=True)

    ranks = []
    for i in range(N):
        correct_mask = labels[sorted_indices[i]] == labels[i]

        rank = torch.nonzero(correct_mask, as_tuple=False)[0].item() + 1
        ranks.append(rank)
    ranks = torch.tensor(ranks, device=x.device, dtype=torch.float)

    metrics = {}
    metrics['R1'] = (ranks <= 1).sum().item() * 100 / N
    metrics['R5'] = (ranks <= 5).sum().item() * 100 / N
    metrics['R10'] = (ranks <= 10).sum().item() * 100 / N
    metrics['MR'] = ranks.median().item()
    metrics["MedianR"] = metrics['MR']
    metrics["MeanR"] = ranks.mean().item()
    metrics["cols"] = ranks
    return metrics
