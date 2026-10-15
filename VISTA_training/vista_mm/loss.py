"""Response-normalized answer supervision."""

import torch
import torch.nn.functional as F


def response_statistics(logits, labels):
    """Per-example response NLL and mean target-token probability, in chunks."""
    pred, target = logits[:, :-1], labels[:, 1:]
    valid = target.ne(-100)
    if not valid.any(1).all():
        raise ValueError("Example has no supervised response tokens")
    total = torch.zeros(len(labels), device=logits.device, dtype=torch.float32)
    probability = torch.zeros_like(total)
    for start in range(0, pred.shape[1], 128):
        truth = target[:, start : start + 128]
        losses = F.cross_entropy(
            pred[:, start : start + 128].float().reshape(-1, pred.shape[-1]),
            truth.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).view(len(labels), -1)
        total = total + losses.sum(1)
        probability = probability + (losses.detach().neg().exp() * truth.ne(-100)).sum(
            1
        )
    return total / valid.sum(1), probability / valid.sum(1)
