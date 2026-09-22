"""Greedy and beam-search text generation implemented directly on causal-LM logits."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def greedy_search(model, input_ids: torch.Tensor, *, max_length: int = 50, eos_token_id: int | None = None):
    sequence = input_ids.clone()
    score = 0.0

    while sequence.shape[1] < max_length:
        logits = model(sequence).logits[:, -1, :]
        log_probs = F.log_softmax(logits, dim=-1)
        next_token = torch.argmax(log_probs, dim=-1, keepdim=True)
        score += float(log_probs.gather(-1, next_token).item())
        sequence = torch.cat([sequence, next_token], dim=-1)
        if eos_token_id is not None and int(next_token.item()) == eos_token_id:
            break

    return sequence, score


def beam_search(
    model,
    input_ids: torch.Tensor,
    *,
    beam_size: int = 3,
    max_length: int = 50,
    eos_token_id: int | None = None,
):
    beams = [(input_ids.clone(), 0.0)]

    while beams[0][0].shape[1] < max_length:
        candidates = []
        all_finished = True

        for sequence, score in beams:
            if eos_token_id is not None and int(sequence[0, -1].item()) == eos_token_id:
                candidates.append((sequence, score))
                continue

            all_finished = False
            logits = model(sequence).logits[:, -1, :]
            log_probs = F.log_softmax(logits, dim=-1)
            values, indices = torch.topk(log_probs, beam_size, dim=-1)

            for index in range(beam_size):
                token = indices[:, index].unsqueeze(-1)
                candidate = torch.cat([sequence, token], dim=-1)
                candidate_score = score + float(values[:, index].item())
                candidates.append((candidate, candidate_score))

        if all_finished:
            break
        beams = sorted(candidates, key=lambda item: item[1], reverse=True)[:beam_size]

    return beams[0]


def beam_search_no_repeat(
    model,
    input_ids: torch.Tensor,
    *,
    beam_size: int = 3,
    max_length: int = 50,
    recent_token_window: int = 20,
    eos_token_id: int | None = None,
):
    """Beam search that blocks tokens used in the recent context window."""
    beams = [(input_ids.clone(), 0.0)]

    while beams[0][0].shape[1] < max_length:
        candidates = []
        all_finished = True

        for sequence, score in beams:
            if eos_token_id is not None and int(sequence[0, -1].item()) == eos_token_id:
                candidates.append((sequence, score))
                continue

            all_finished = False
            logits = model(sequence).logits[:, -1, :]
            log_probs = F.log_softmax(logits, dim=-1)
            recent = sequence[0, -recent_token_window:].unique()
            log_probs[:, recent] = torch.finfo(log_probs.dtype).min
            values, indices = torch.topk(log_probs, beam_size, dim=-1)

            for index in range(beam_size):
                token = indices[:, index].unsqueeze(-1)
                candidate = torch.cat([sequence, token], dim=-1)
                candidate_score = score + float(values[:, index].item())
                candidates.append((candidate, candidate_score))

        if all_finished:
            break
        beams = sorted(candidates, key=lambda item: item[1], reverse=True)[:beam_size]

    return beams[0]
