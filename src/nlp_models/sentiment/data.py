"""Preprocessing helpers for aspect-based sentiment analysis."""

POLARITY_TO_ID = {"negative": 0, "neutral": 1, "positive": 2}


def combine_review_and_aspect(review: str, aspect: str) -> str:
    """Create the sentence/aspect input used by the DistilBERT models."""
    return f"{review} [SEP] {aspect}"


def polarity_to_id(label: str) -> int:
    try:
        return POLARITY_TO_ID[label]
    except KeyError as exc:
        raise ValueError(f"Unknown polarity: {label}") from exc
