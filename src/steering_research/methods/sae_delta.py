from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from steering_research.data.schema import ContrastPair
from steering_research.models.base import ActivationBackend, ActivationRequest


@dataclass(frozen=True)
class SaeDeltaResult:
    feature_index: int
    delta: float
    positive_mean: float
    negative_mean: float


def _fake_sparse_encode(vector: np.ndarray, width: int = 4096, top_k: int = 50) -> np.ndarray:
    seed = int(abs(float(vector.sum())) * 1_000_000) % (2**32)
    rng = np.random.default_rng(seed)
    projection = rng.normal(0.0, 1.0, (width, vector.shape[0]))
    acts = projection @ vector
    top_idx = np.argpartition(acts, -top_k)[-top_k:]
    sparse = np.zeros(width, dtype=np.float64)
    sparse[top_idx] = acts[top_idx]
    return sparse


def rank_sae_deltas(
    backend: ActivationBackend,
    pairs: list[ContrastPair],
    request: ActivationRequest,
    top_features: int = 25,
    sae: Any | None = None,
) -> list[SaeDeltaResult]:
    pos_acts: list[np.ndarray] = []
    neg_acts: list[np.ndarray] = []
    for pair in pairs:
        pos = backend.activation(pair.positive, request)
        neg = backend.activation(pair.negative, request)
        if sae is None:
            pos_acts.append(_fake_sparse_encode(pos))
            neg_acts.append(_fake_sparse_encode(neg))
        else:
            pos_acts.append(np.asarray(sae.encode_numpy(pos), dtype=np.float64))
            neg_acts.append(np.asarray(sae.encode_numpy(neg), dtype=np.float64))
    pos_mean = np.stack(pos_acts, axis=0).mean(axis=0)
    neg_mean = np.stack(neg_acts, axis=0).mean(axis=0)
    delta = pos_mean - neg_mean
    top_idx = np.argsort(np.abs(delta))[-top_features:][::-1]
    return [
        SaeDeltaResult(
            feature_index=int(idx),
            delta=float(delta[idx]),
            positive_mean=float(pos_mean[idx]),
            negative_mean=float(neg_mean[idx]),
        )
        for idx in top_idx
    ]
