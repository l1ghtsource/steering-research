from __future__ import annotations

from pathlib import Path

from steering_research.data import BenchmarkStore
from steering_research.methods import build_direction, evaluate_direction, rank_sae_deltas
from steering_research.models import FakeActivationBackend
from steering_research.models.base import ActivationRequest


def test_direction_and_sae_delta_on_real_benchmark() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    store = BenchmarkStore.from_repo_root(repo_root)
    pairs = store.pairs(
        behavior="hallucination",
        origin_bucket="source_backed_contrasts",
        limit=8,
    )
    backend = FakeActivationBackend(hidden_size=64)
    request = ActivationRequest(layer=0, activation_view="last_prompt_token")
    direction = build_direction(backend, pairs[:5], request)
    metrics = evaluate_direction(backend, pairs[5:], direction)
    assert direction.direction.shape == (64,)
    assert metrics["n_eval_pairs"] == 3.0
    deltas = rank_sae_deltas(backend, pairs[:4], request, top_features=5)
    assert len(deltas) == 5
