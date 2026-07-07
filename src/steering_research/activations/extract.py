from __future__ import annotations

import numpy as np

from steering_research.data.schema import ContrastPair
from steering_research.models.base import ActivationBackend, ActivationRequest


def extract_pair_matrix(
    backend: ActivationBackend,
    pairs: list[ContrastPair],
    request: ActivationRequest,
) -> tuple[np.ndarray, np.ndarray]:
    positives = [backend.activation(pair.positive, request) for pair in pairs]
    negatives = [backend.activation(pair.negative, request) for pair in pairs]
    return np.stack(positives, axis=0), np.stack(negatives, axis=0)
