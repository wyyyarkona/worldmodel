from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PairwiseScoreLoss(nn.Module):
    # Weighted hard-label BCE:
    # a better than b -> 1
    # b better than or equal to a -> 0
    # Keeps the old constructor signature so existing training scripts/configs
    # do not need to change when switching from soft labels to standard BCE.
    def __init__(self, tau=0.3, margin=0.05):
        super().__init__()
        self.tau = tau
        self.margin = margin

    def forward(self, logits, score_a, score_b, sample_weight=None):
        target = (score_a > score_b).to(dtype=logits.dtype)
        if sample_weight is None:
            weight = torch.ones_like(target, dtype=logits.dtype)
        else:
            weight = sample_weight.to(dtype=logits.dtype)
        loss = F.binary_cross_entropy_with_logits(
            logits,
            target,
            weight=weight,
            reduction="sum",
        )
        normalizer = torch.clamp(weight.sum(), min=1e-6)
        return loss / normalizer


class WarmupAlignmentLoss(nn.Module):
    # Auxiliary projector-statistics loss used only in the warmup stage.
    def __init__(self, target_mean=None, target_std=None, weight=0.1):
        super().__init__()
        self.weight = weight
        self.register_buffer(
            "target_mean",
            torch.zeros(1) if target_mean is None else torch.as_tensor(target_mean),
            persistent=False,
        )
        self.register_buffer(
            "target_std",
            torch.ones(1) if target_std is None else torch.as_tensor(target_std),
            persistent=False,
        )

    def forward(self, proj_mean, proj_std):
        mean_target = self.target_mean.to(device=proj_mean.device, dtype=proj_mean.dtype)
        std_target = self.target_std.to(device=proj_std.device, dtype=proj_std.dtype)
        # A scalar target is expanded to all hidden dimensions for the simple N(0,1)
        # placeholder case described in the plan.
        if mean_target.numel() == 1:
            mean_target = mean_target.expand_as(proj_mean)
        if std_target.numel() == 1:
            std_target = std_target.expand_as(proj_std)
        loss = F.mse_loss(proj_mean, mean_target) + F.mse_loss(proj_std, std_target)
        return self.weight * loss
