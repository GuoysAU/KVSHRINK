"""Evaluation metrics and scoring functions."""

from .metrics import evaluate_predictions
from .scoring import (
    score_multiple_choice,
    score_per_option_context,
)

__all__ = [
    "evaluate_predictions",
    "score_multiple_choice",
    "score_per_option_context",
]
