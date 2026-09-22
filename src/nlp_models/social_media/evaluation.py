"""Evaluation helpers for humour-rating models."""

from __future__ import annotations

import numpy as np


def normalize_rating(values, maximum: float = 5.0):
    return np.asarray(values, dtype=float) / maximum


def average_predictions(prediction_runs) -> np.ndarray:
    values = np.asarray(prediction_runs, dtype=float)
    if values.ndim < 2:
        raise ValueError("prediction_runs must contain predictions from at least one model.")
    return values.mean(axis=0)


def mean_squared_error(targets, predictions) -> float:
    targets = np.asarray(targets, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    return float(np.mean((targets - predictions) ** 2))
