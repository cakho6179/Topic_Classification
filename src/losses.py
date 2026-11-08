"""
Advanced Loss Functions for Imbalanced 13-Class Topic Classification.
Includes Weighted Cross Entropy, Multi-Class Focal Loss, and Class-Balanced Loss.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Sequence

class WeightedCrossEntropyLoss(nn.Module):
    """
    Cross Entropy Loss with inverse-frequency class weights.
    """
    def __init__(self, class_weights: Optional[Sequence[float]] = None):
        super().__init__()
        if class_weights is not None:
            self.register_buffer("weight", torch.tensor(class_weights, dtype=torch.float32))
        else:
            self.weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, targets, weight=self.weight)

class MultiClassFocalLoss(nn.Module):
    """
    Multi-Class Focal Loss (Lin et al., ICCV 2017).
    Loss = - alpha_t * (1 - p_t)^gamma * log(p_t)
    Down-weights well-classified examples and focuses gradients on hard edge-cases.
    """
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[Sequence[float]] = None,
        reduction: str = "mean"
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (B, C), targets: (B,)
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        probs = torch.softmax(logits, dim=-1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1) # p for true class
        
        focal_weight = (1.0 - pt) ** self.gamma

        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * focal_weight * ce_loss
        else:
            focal_loss = focal_weight * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss

class ClassBalancedFocalLoss(nn.Module):
    """
    Class-Balanced Focal Loss based on Effective Number of Samples (Cui et al., CVPR 2019).
    E_n = (1 - beta^n) / (1 - beta)
    weight_c = (1 - beta) / (1 - beta^{n_c})
    """
    def __init__(
        self,
        samples_per_class: Sequence[int],
        beta: float = 0.999,
        gamma: float = 2.0,
        reduction: str = "mean"
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

        # Compute effective weights
        effective_num = [1.0 - (beta ** count) if count > 0 else 1.0 for count in samples_per_class]
        weights = [(1.0 - beta) / max(en, 1e-8) for en in effective_num]
        
        # Normalize weights so sum equals number of classes
        total_w = sum(weights)
        num_classes = len(samples_per_class)
        norm_weights = [w * num_classes / total_w for w in weights]
        
        self.register_buffer("class_weights", torch.tensor(norm_weights, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        probs = torch.softmax(logits, dim=-1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        focal_weight = (1.0 - pt) ** self.gamma
        alpha_t = self.class_weights.gather(0, targets)
        
        loss = alpha_t * focal_weight * ce_loss
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

def create_loss_function(
    loss_name: str,
    samples_per_class: Optional[Sequence[int]] = None,
    gamma: float = 2.0,
    beta: float = 0.999
) -> nn.Module:
    """
    Factory function to instantiate loss functions by name.
    """
    loss_name = loss_name.lower()
    if loss_name == "cross_entropy":
        return nn.CrossEntropyLoss()
    elif loss_name == "weighted_ce":
        if samples_per_class is None:
            raise ValueError("samples_per_class is required for weighted_ce")
        total = sum(samples_per_class)
        k = len(samples_per_class)
        weights = [total / (k * max(c, 1)) for c in samples_per_class]
        return WeightedCrossEntropyLoss(weights)
    elif loss_name == "focal":
        return MultiClassFocalLoss(gamma=gamma)
    elif loss_name == "class_balanced":
        if samples_per_class is None:
            raise ValueError("samples_per_class is required for class_balanced")
        return ClassBalancedFocalLoss(samples_per_class, beta=beta, gamma=gamma)
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")
