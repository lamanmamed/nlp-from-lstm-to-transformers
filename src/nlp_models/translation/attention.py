"""Luong-style dot-product attention."""

from __future__ import annotations

import torch
from torch import nn


class LuongAttention(nn.Module):
    """Combine decoder states with source-side context vectors."""

    def forward(
        self,
        encoder_outputs: torch.Tensor,
        decoder_outputs: torch.Tensor,
        source_mask: torch.Tensor | None = None,
        *,
        return_weights: bool = False,
    ):
        # [batch, source_len, hidden] x [batch, hidden, target_len]
        scores = torch.bmm(encoder_outputs, decoder_outputs.transpose(1, 2))

        if source_mask is not None:
            mask = source_mask.bool().unsqueeze(-1)
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

        weights = torch.softmax(scores, dim=1).transpose(1, 2)
        context = torch.bmm(weights, encoder_outputs)
        combined = torch.cat([decoder_outputs, context], dim=-1)

        if return_weights:
            return combined, weights
        return combined
