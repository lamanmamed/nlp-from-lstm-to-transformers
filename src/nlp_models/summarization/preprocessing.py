"""Preprocessing helpers for T5 summarization and keyword-conditioned generation."""

from __future__ import annotations

import random

import numpy as np


def add_summarization_prefix(text: str) -> str:
    return "summarize: " + text


def sample_keywords(text: str, *, count: int = 20, rng: random.Random | None = None) -> str:
    words = text.split()
    if not words:
        return ""
    rng = rng or random
    selected = rng.sample(words, min(count, len(words)))
    return " ".join(selected)


def token_count_stats(texts) -> tuple[float, float]:
    counts = np.asarray([len(text.split()) for text in texts], dtype=float)
    if counts.size == 0:
        return 0.0, 0.0
    return float(counts.mean()), float(counts.std())
