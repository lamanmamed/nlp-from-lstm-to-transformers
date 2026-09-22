"""Pooling functions shared by DistilBERT classifiers and regressors."""

from __future__ import annotations

import torch


def masked_mean(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
    total = (hidden_states * mask).sum(dim=1)
    count = mask.sum(dim=1).clamp_min(1.0)
    return total / count
