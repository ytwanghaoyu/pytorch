import torch

from .activation import sigmoid
from .functional import logsumexp


def binary_cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = sigmoid(logits)
    loss = -(
        target * torch.log(pred + 1e-12) + (1 - target) * torch.log(1 - pred + 1e-12)
    )
    return loss.mean()


def binary_cross_entropy_logits(
    logits: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    loss = (
        torch.clamp(logits, min=0)
        - logits * target
        + torch.log1p(torch.exp(-torch.abs(logits)))
    )
    return loss.mean()


def cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    log_sum_exp = logsumexp(logits, dim=1)

    # log softmax
    log_probs = logits - log_sum_exp

    target_indices = target.unsqueeze(1)
    gathered_log_probs = torch.gather(log_probs, dim=1, index=target_indices)

    loss = -gathered_log_probs.squeeze(1)
    return loss.mean()


if __name__ == "__main__":
    bce_gt = torch.tensor
