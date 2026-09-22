"""Small data helpers for the Vietnamese-English translation experiment."""

from __future__ import annotations

from collections import Counter

import numpy as np


class Vocabulary:
    def __init__(self, sentences, min_count: int = 11, extra_tokens=()):
        counts = Counter(token.lower() for sentence in sentences for token in sentence)
        self.tokens = ["<pad>", "<unk>", *extra_tokens]
        existing = set(self.tokens)
        self.tokens.extend(
            token for token, count in counts.items()
            if count >= min_count and token not in existing
        )
        self.token_to_id = {token: index for index, token in enumerate(self.tokens)}
        self.pad_id = self.token_to_id["<pad>"]
        self.unk_id = self.token_to_id["<unk>"]

    def encode(self, sentence):
        return [self.token_to_id.get(token.lower(), self.unk_id) for token in sentence]


def pad_sequences(sequences, max_length: int | None = None, pad_value: int = 0):
    if not sequences:
        return np.empty((0, 0), dtype=np.int64)
    if max_length is None:
        max_length = max(len(sequence) for sequence in sequences)
    result = np.full((len(sequences), max_length), pad_value, dtype=np.int64)
    for row, sequence in enumerate(sequences):
        clipped = sequence[:max_length]
        result[row, : len(clipped)] = clipped
    return result
