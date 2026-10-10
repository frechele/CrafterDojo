import torch


def compute_metrics(
    x: torch.Tensor,
    labels: torch.Tensor | None = None,
    gt_indices: torch.Tensor | None = None,
):
    N = x.shape[0]
    sorted_indices = torch.argsort(x, dim=1, descending=True)

    if gt_indices is not None:
        if labels is not None:
            raise ValueError("Provide only one of labels or gt_indices.")
        if gt_indices.shape[0] != N:
            raise ValueError("gt_indices must have the same length as rows.")
        if gt_indices.min().item() < 0 or gt_indices.max().item() >= x.shape[1]:
            raise ValueError("gt_indices must be within [0, num_texts).")
        matches = sorted_indices == gt_indices.view(-1, 1)
        if not torch.all(matches.any(dim=1)):
            missing = torch.nonzero(
                ~matches.any(dim=1), as_tuple=False
            ).squeeze(-1)
            raise ValueError(
                f"gt_indices not found for rows: {missing.tolist()}"
            )
        ranks = matches.float().argmax(dim=1) + 1
    else:
        if labels is None:
            raise ValueError("labels is required when gt_indices is None.")
        if x.shape[0] != x.shape[1]:
            raise ValueError("Use gt_indices for non-square similarity matrices.")
        if labels.shape[0] != x.shape[1]:
            raise ValueError("labels must match the number of columns.")

        ranks = []
        for i in range(N):
            correct_mask = labels[sorted_indices[i]] == labels[i]
            rank = torch.nonzero(correct_mask, as_tuple=False)[0].item() + 1
            ranks.append(rank)
        ranks = torch.tensor(ranks, device=x.device, dtype=torch.float)

    metrics = {}
    metrics["R1"] = (ranks <= 1).sum().item() * 100 / N
    metrics["R5"] = (ranks <= 5).sum().item() * 100 / N
    metrics["R10"] = (ranks <= 10).sum().item() * 100 / N
    metrics["MR"] = ranks.median().item()
    metrics["MedianR"] = metrics["MR"]
    metrics["MeanR"] = ranks.float().mean().item()
    metrics["cols"] = ranks
    return metrics
