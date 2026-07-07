from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return math.nan
    return numerator / denominator


def mean(values: Sequence[float]) -> float:
    if not values:
        return math.nan
    return float(sum(values) / len(values))


def accuracy(predicted: Sequence[bool], expected: Sequence[bool]) -> float:
    if len(predicted) != len(expected):
        msg = "predicted and expected must have the same length"
        raise ValueError(msg)
    if not predicted:
        return math.nan
    return sum(int(a == b) for a, b in zip(predicted, expected, strict=True)) / len(predicted)


def direction_accuracy(scores: Sequence[float], margin: float = 0.0) -> float:
    if not scores:
        return math.nan
    return sum(int(score > margin) for score in scores) / len(scores)


def auroc(scores: Sequence[float], labels: Sequence[int]) -> float:
    if len(scores) != len(labels):
        msg = "scores and labels must have the same length"
        raise ValueError(msg)
    positives = [s for s, y in zip(scores, labels, strict=True) if y == 1]
    negatives = [s for s, y in zip(scores, labels, strict=True) if y == 0]
    if not positives or not negatives:
        return math.nan
    wins = 0.0
    total = 0
    for pos in positives:
        for neg in negatives:
            total += 1
            if pos > neg:
                wins += 1.0
            elif pos == neg:
                wins += 0.5
    return wins / total


def unit_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return vector.copy()
    return vector / norm
