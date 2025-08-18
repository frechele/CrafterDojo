import torch
import torch.nn.functional as F


def focal_loss(logits: torch.Tensor, labels: torch.Tensor, reduce: bool=True, pos_alpha: float=0.75, gamma: float=2.0):
    labels = labels.float()
    bce_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    
    p = torch.sigmoid(logits)
    pt = torch.where(labels == 1, p, 1 - p)
    
    alpha = torch.where(labels == 1, pos_alpha, 1-pos_alpha)
    
    loss = alpha * ((1 - pt) ** gamma) * bce_loss
    if reduce:
        return loss.mean()

    return loss
