from __future__ import annotations

from pathlib import Path

from steering_research.data import BenchmarkStore


def test_latent_behavior_bench_submodule_loads() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    store = BenchmarkStore.from_repo_root(repo_root)
    result = store.validate()
    assert result["examples"] == 8688
    assert result["contrasts"] == 1000
    assert result["missing_positive_refs"] == 0
    assert result["missing_negative_refs"] == 0


def test_clean_split_source_backed_pairs() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    store = BenchmarkStore.from_repo_root(repo_root)
    pairs = store.pairs(
        behavior="sycophancy",
        origin_bucket="source_backed_contrasts",
        limit=3,
    )
    assert len(pairs) == 3
    assert all(pair.contrast.behavior == "sycophancy" for pair in pairs)
    assert all(pair.positive.id != pair.negative.id for pair in pairs)
