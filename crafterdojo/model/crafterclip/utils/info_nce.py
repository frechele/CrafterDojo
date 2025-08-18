import torch
import torch.nn.functional as F


def multi_positive_info_nce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    logits = logits / temperature

    alpha, _ = logits.max(dim=1, keepdim=True)
    logits = logits - alpha

    exp_logits = torch.exp(logits)
    sum_exp = exp_logits.sum(dim=1, keepdim=True)

    pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    pos_exp = (exp_logits * pos_mask).sum(dim=1)

    eps = 1e-8
    log_prob = torch.log(pos_exp + eps) - torch.log(sum_exp.squeeze(1) + eps)

    return -log_prob.mean()


def info_nce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    loss_1 = multi_positive_info_nce(logits, labels, temperature)
    loss_2 = multi_positive_info_nce(logits.T, labels, temperature)

    return loss_1 + loss_2
