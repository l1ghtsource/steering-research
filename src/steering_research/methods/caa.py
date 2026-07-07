from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from steering_research.data.schema import ContrastPair
from steering_research.metrics.basic import direction_accuracy, unit_vector
from steering_research.models.base import ActivationBackend, ActivationRequest, FloatArray


@dataclass(frozen=True)
class DirectionResult:
    behavior: str
    layer: int
    activation_view: str
    component: str
    n_pairs: int
    direction: FloatArray
    unit_direction: FloatArray


def build_direction(
    backend: ActivationBackend,
    pairs: list[ContrastPair],
    request: ActivationRequest,
) -> DirectionResult:
    if not pairs:
        msg = "Cannot build direction from zero pairs"
        raise ValueError(msg)
    diffs: list[FloatArray] = []
    for pair in pairs:
        pos = backend.activation(pair.positive, request)
        neg = backend.activation(pair.negative, request)
        diffs.append(pos - neg)
    direction = np.stack(diffs, axis=0).mean(axis=0)
    return DirectionResult(
        behavior=pairs[0].contrast.behavior,
        layer=request.layer,
        activation_view=request.activation_view,
        component=request.component,
        n_pairs=len(pairs),
        direction=direction,
        unit_direction=unit_vector(direction),
    )


def evaluate_direction(
    backend: ActivationBackend,
    pairs: list[ContrastPair],
    direction: DirectionResult,
) -> dict[str, float]:
    request = ActivationRequest(
        layer=direction.layer,
        activation_view=direction.activation_view,
        component=direction.component,
    )
    scores: list[float] = []
    pos_scores: list[float] = []
    neg_scores: list[float] = []
    for pair in pairs:
        pos = backend.activation(pair.positive, request)
        neg = backend.activation(pair.negative, request)
        pos_score = float(pos @ direction.unit_direction)
        neg_score = float(neg @ direction.unit_direction)
        pos_scores.append(pos_score)
        neg_scores.append(neg_score)
        scores.append(pos_score - neg_score)
    margin_scores = [abs(score) for score in scores]
    return {
        "n_eval_pairs": float(len(pairs)),
        "direction_accuracy": direction_accuracy(scores),
        "mean_projection_gap": float(np.mean(scores)) if scores else float("nan"),
        "median_projection_gap": float(np.median(scores)) if scores else float("nan"),
        "mean_abs_margin": float(np.mean(margin_scores)) if margin_scores else float("nan"),
        "mean_positive_projection": float(np.mean(pos_scores)) if pos_scores else float("nan"),
        "mean_negative_projection": float(np.mean(neg_scores)) if neg_scores else float("nan"),
    }
