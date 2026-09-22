"""Batch preprocessing for T5 summarization and keyword-to-text generation."""

from __future__ import annotations

import random

from .preprocessing import add_summarization_prefix, sample_keywords


def preprocess_summaries(samples, tokenizer, *, max_document_length: int = 1024, max_summary_length: int = 128):
    inputs = [add_summarization_prefix(text) for text in samples["document"]]
    model_inputs = tokenizer(inputs, max_length=max_document_length, truncation=True)
    targets = tokenizer(
        text_target=samples["summary"],
        max_length=max_summary_length,
        truncation=True,
    )
    model_inputs["labels"] = targets["input_ids"]
    return model_inputs


def preprocess_keyword_generation(
    samples,
    tokenizer,
    *,
    keyword_count: int = 20,
    max_document_length: int = 1024,
    max_summary_length: int = 128,
    seed: int | None = None,
):
    rng = random.Random(seed)
    inputs = [
        sample_keywords(summary, count=keyword_count, rng=rng)
        for summary in samples["summary"]
    ]
    model_inputs = tokenizer(inputs, max_length=max_document_length, truncation=True)
    targets = tokenizer(
        text_target=samples["summary"],
        max_length=max_summary_length,
        truncation=True,
    )
    model_inputs["labels"] = targets["input_ids"]
    return model_inputs
