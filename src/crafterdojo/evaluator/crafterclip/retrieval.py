"""Shared retrieval metrics and text-normalization helpers for CraftCLIP evaluators."""

import torch


def normalize_texts_join(texts):
    """Flatten list-of-lists into single strings by whitespace-joining."""
    normalized = []
    for text in texts:
        if isinstance(text, (list, tuple)):
            normalized.append(" ".join(str(t) for t in text))
        elif not isinstance(text, str):
            normalized.append(str(text))
        else:
            normalized.append(text)
    return normalized


def normalize_texts_first(texts):
    """Take first element of each list (fallback for single-string ground truths)."""
    normalized = []
    for text in texts:
        if isinstance(text, list):
            normalized.append(text[0] if text else "")
        else:
            normalized.append(text)
    return normalized


def compute_retrieval_metrics(similarity_matrix, gt_indices=None):
    """Compute retrieval metrics (R@1, R@5, R@10, MedR, MeanR) both directions."""
    if gt_indices is None:
        gt_indices = torch.arange(
            similarity_matrix.size(0), device=similarity_matrix.device
        )
    if not torch.is_tensor(gt_indices):
        gt_indices = torch.as_tensor(gt_indices, device=similarity_matrix.device)
    gt_indices = gt_indices.to(device=similarity_matrix.device, dtype=torch.long)
    if gt_indices.shape[0] != similarity_matrix.size(0):
        raise ValueError("gt_indices must have the same length as rows.")
    if (
        gt_indices.min().item() < 0
        or gt_indices.max().item() >= similarity_matrix.size(1)
    ):
        raise ValueError("gt_indices must be within [0, num_texts).")

    sorted_indices = torch.argsort(similarity_matrix, dim=1, descending=True)
    matches = sorted_indices == gt_indices.view(-1, 1)
    v2t_ranks = matches.float().argmax(dim=1) + 1

    t2v_ranks_list = []
    for text_idx in torch.unique(gt_indices).tolist():
        sims = similarity_matrix[:, text_idx]
        sorted_video_indices = torch.argsort(sims, descending=True)
        gt_video_indices = torch.nonzero(
            gt_indices == text_idx, as_tuple=False
        ).squeeze(-1)
        matches = (
            sorted_video_indices.view(-1, 1) == gt_video_indices.view(1, -1)
        ).any(dim=1)
        rank = torch.nonzero(matches, as_tuple=False)[0].item() + 1
        t2v_ranks_list.append(rank)

    t2v_ranks = (
        torch.tensor(t2v_ranks_list, device=similarity_matrix.device)
        if t2v_ranks_list
        else v2t_ranks
    )

    def recall_metrics(ranks):
        return (
            (ranks <= 1).float().mean().item(),
            (ranks <= 5).float().mean().item(),
            (ranks <= 10).float().mean().item(),
            ranks.float().median().item(),
            ranks.float().mean().item(),
        )

    v2t_r1, v2t_r5, v2t_r10, v2t_medr, v2t_meanr = recall_metrics(v2t_ranks)
    t2v_r1, t2v_r5, t2v_r10, t2v_medr, t2v_meanr = recall_metrics(t2v_ranks)

    return {
        "v2t_r1": v2t_r1, "v2t_r5": v2t_r5, "v2t_r10": v2t_r10,
        "v2t_medr": v2t_medr, "v2t_meanr": v2t_meanr,
        "t2v_r1": t2v_r1, "t2v_r5": t2v_r5, "t2v_r10": t2v_r10,
        "t2v_medr": t2v_medr, "t2v_meanr": t2v_meanr,
    }
