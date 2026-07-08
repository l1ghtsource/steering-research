from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, TypeVar

import numpy as np

from steering_research.cli.backend import build_backend
from steering_research.data import BenchmarkStore
from steering_research.methods import (
    DirectionResult,
    build_direction,
    evaluate_direction,
    rank_sae_deltas,
)
from steering_research.metrics.basic import auroc, unit_vector
from steering_research.models.base import ActivationRequest
from steering_research.reports.dashboard import write_static_dashboard
from steering_research.reports.markdown import write_experiment_report
from steering_research.runtime import RunLogger, load_yaml, resolve_path
from steering_research.scoring import score_generation
from steering_research.steering import run_steering_sweep

T = TypeVar("T")


def _split_train_eval(items: list[T], fraction: float) -> tuple[list[T], list[T]]:
    pivot = (
        max(1, min(len(items) - 1, int(len(items) * fraction))) if len(items) > 1 else len(items)
    )
    return items[:pivot], items[pivot:]


def _as_float(value: object) -> float:
    if isinstance(value, int | float | str):
        return float(value)
    msg = f"Cannot convert value to float: {value!r}"
    raise TypeError(msg)


def _load_benchmark(repo_root: Path, dataset_config: str) -> BenchmarkStore:
    cfg = load_yaml(resolve_path(dataset_config, repo_root))
    return BenchmarkStore(
        root=resolve_path(str(cfg["root"]), repo_root),
        examples_path=str(cfg.get("examples", "processed/examples.jsonl")),
        contrasts_path=str(cfg.get("contrasts", "processed/contrasts.jsonl")),
        clean_splits_path=str(cfg.get("clean_splits", "processed/eval_splits_clean.json")),
    )


def _direction_from_vector(base: DirectionResult, vector: np.ndarray) -> DirectionResult:
    return DirectionResult(
        behavior=base.behavior,
        layer=base.layer,
        activation_view=base.activation_view,
        component=base.component,
        n_pairs=base.n_pairs,
        direction=vector,
        unit_direction=unit_vector(vector),
    )


def _negated_direction(direction: DirectionResult) -> DirectionResult:
    return _direction_from_vector(direction, -direction.direction)


def _orthogonalize(vector: np.ndarray, controls: list[np.ndarray]) -> np.ndarray:
    result = vector.astype(np.float64).copy()
    for control in controls:
        unit = unit_vector(control.astype(np.float64))
        result = result - float(result @ unit) * unit
    return result


def _aggregate_marker_rows(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[object, ...], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[key] for key in keys), []).append(row)
    aggregate_rows: list[dict[str, Any]] = []
    for key_values, group in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        out = dict(zip(keys, key_values, strict=True))
        out.update(
            {
                "n": len(group),
                "mean_refusal_marker": float(
                    np.mean([_as_float(row["refusal_marker"]) for row in group])
                ),
                "mean_agreement_marker": float(
                    np.mean([_as_float(row["agreement_marker"]) for row in group])
                ),
                "mean_uncertainty_marker": float(
                    np.mean([_as_float(row["uncertainty_marker"]) for row in group])
                ),
                "mean_unsafe_planning_marker": float(
                    np.mean([_as_float(row["unsafe_planning_marker"]) for row in group])
                ),
                "mean_length_tokens": float(
                    np.mean([_as_float(row["length_tokens"]) for row in group])
                ),
                "mean_repetition_proxy": float(
                    np.mean([_as_float(row["repetition_proxy"]) for row in group])
                ),
            }
        )
        aggregate_rows.append(out)
    return aggregate_rows


def _write_generation_outputs(
    logger: RunLogger,
    rows: list[dict[str, Any]],
    aggregate_rows: list[dict[str, Any]],
    summary: dict[str, Any],
    title: str,
) -> None:
    with (logger.run_dir / "tables" / "generations.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fieldnames = sorted({key for row in rows for key in row}) if rows else ["empty"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.write_json("summary.json", summary)
    logger.write_json("aggregate.json", {"rows": aggregate_rows})
    write_experiment_report(logger.run_dir / "report.md", title, summary, aggregate_rows[:100])
    write_static_dashboard(logger.run_dir.parent)


def _generate_vector_sweep(
    backend: Any,
    examples: list[Any],
    layer: int,
    vector: np.ndarray,
    alphas: list[float],
    max_new_tokens: int,
    batch_size: int,
    logger: RunLogger,
    extra: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    extra = extra or {}
    generate_batch = getattr(backend, "generate_batch", None)
    for alpha in alphas:
        steering = (layer, vector, alpha)
        for start in range(0, len(examples), max(1, batch_size)):
            batch = examples[start : start + max(1, batch_size)]
            if callable(generate_batch) and batch_size > 1:
                generations = generate_batch(
                    batch, max_new_tokens=max_new_tokens, steering=steering
                )
            else:
                generations = [
                    backend.generate(example, max_new_tokens=max_new_tokens, steering=steering)
                    for example in batch
                ]
            for example, generation in zip(batch, generations, strict=True):
                row = {
                    **extra,
                    "example_id": example.id,
                    "behavior": ",".join(example.behavior_axes),
                    "alpha": alpha,
                    "text": generation.text,
                    **score_generation(generation.text),
                }
                rows.append(row)
                logger.log_metric(row)
    return rows


def _generate_multi_steering_sweep(
    backend: Any,
    examples: list[Any],
    steering_groups: list[dict[str, Any]],
    alphas: list[float],
    max_new_tokens: int,
    batch_size: int,
    logger: RunLogger,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    generate_multi = getattr(backend, "generate_batch_multi_steering", None)
    for group in steering_groups:
        name = str(group["name"])
        hooks = [
            (int(item["layer"]), np.asarray(item["vector"], dtype=np.float64))
            for item in group["hooks"]
        ]
        for alpha in alphas:
            alpha_scale = 1.0 / max(1, len(hooks)) if bool(group.get("divide_alpha", True)) else 1.0
            steerings = [(layer, vector, alpha * alpha_scale) for layer, vector in hooks]
            for start in range(0, len(examples), max(1, batch_size)):
                batch = examples[start : start + max(1, batch_size)]
                if callable(generate_multi) and batch_size > 1:
                    generations = generate_multi(batch, steerings, max_new_tokens=max_new_tokens)
                else:
                    generations = [
                        backend.generate(
                            example,
                            max_new_tokens=max_new_tokens,
                            steering=steerings[0] if steerings else None,
                        )
                        for example in batch
                    ]
                for example, generation in zip(batch, generations, strict=True):
                    row = {
                        "variant": name,
                        "n_hooks": len(steerings),
                        "example_id": example.id,
                        "behavior": ",".join(example.behavior_axes),
                        "alpha": alpha,
                        "text": generation.text,
                        **score_generation(generation.text),
                    }
                    rows.append(row)
                    logger.log_metric(row)
    return rows


def run_e001(repo_root: Path, config_path: Path, backend_kind: str | None = None) -> Path:
    cfg = load_yaml(config_path)
    backend_name = backend_kind or str(cfg.get("backend", "fake"))
    backend = build_backend(backend_name, resolve_path(str(cfg["model"]), repo_root))
    store = _load_benchmark(repo_root, str(cfg["dataset"]))
    logger = RunLogger(
        resolve_path(str(cfg.get("output_dir", "runs")), repo_root),
        str(cfg["name"]),
        {"config": cfg, "backend": backend.name, "benchmark": store.validate()},
    )
    rows: list[dict[str, Any]] = []
    for behavior in cfg["behaviors"]:
        for origin in cfg["origins"]:
            pairs = store.pairs(
                behavior=str(behavior),
                origin_bucket=str(origin),
                limit=int(cfg.get("limit_per_behavior", 24)),
            )
            if len(pairs) < 2:
                continue
            train_pairs, eval_pairs = _split_train_eval(
                pairs, float(cfg.get("train_fraction", 0.7))
            )
            if not eval_pairs:
                eval_pairs = train_pairs
            for layer in cfg["layers"]:
                for activation_view in cfg["activation_views"]:
                    request = ActivationRequest(
                        layer=int(layer),
                        activation_view=str(activation_view),
                        component=str(cfg.get("component", "residual")),
                    )
                    direction = build_direction(backend, train_pairs, request)
                    metrics = evaluate_direction(backend, eval_pairs, direction)
                    row = {
                        "experiment": cfg["name"],
                        "behavior": behavior,
                        "origin": origin,
                        "layer": int(layer),
                        "activation_view": activation_view,
                        "n_train_pairs": len(train_pairs),
                        **metrics,
                    }
                    rows.append(row)
                    logger.log_metric(row)
    best = max(rows, key=lambda row: float(row["direction_accuracy"])) if rows else {}
    summary = {
        "experiment": cfg["name"],
        "backend": backend.name,
        "rows": len(rows),
        "best_behavior": best.get("behavior", ""),
        "best_origin": best.get("origin", ""),
        "best_layer": best.get("layer", ""),
        "best_activation_view": best.get("activation_view", ""),
        "best_direction_accuracy": best.get("direction_accuracy", float("nan")),
    }
    logger.write_json("summary.json", summary)
    write_experiment_report(logger.run_dir / "report.md", "E001 Mean Direction", summary, rows[:50])
    write_static_dashboard(logger.run_dir.parent)
    return logger.run_dir


def run_e002(repo_root: Path, config_path: Path, backend_kind: str | None = None) -> Path:
    cfg = load_yaml(config_path)
    backend_name = backend_kind or str(cfg.get("backend", "fake"))
    backend = build_backend(backend_name, resolve_path(str(cfg["model"]), repo_root))
    store = _load_benchmark(repo_root, str(cfg["dataset"]))
    logger = RunLogger(
        resolve_path(str(cfg.get("output_dir", "runs")), repo_root),
        str(cfg["name"]),
        {"config": cfg, "backend": backend.name, "benchmark": store.validate()},
    )
    train_pairs = store.pairs(
        behavior=str(cfg["behavior"]),
        origin_bucket=str(cfg["origin"]),
        limit=int(cfg.get("train_limit", 16)),
    )
    eval_pairs = store.pairs(
        behavior=str(cfg["behavior"]),
        origin_bucket=str(cfg["origin"]),
        limit=int(cfg.get("eval_limit", 16)),
    )
    request = ActivationRequest(
        layer=int(cfg["layer"]),
        activation_view=str(cfg["activation_view"]),
    )
    direction = build_direction(backend, train_pairs, request)
    scores: list[float] = []
    labels: list[int] = []
    rows: list[dict[str, Any]] = []
    for pair in eval_pairs:
        pos_score = float(backend.activation(pair.positive, request) @ direction.unit_direction)
        neg_score = float(backend.activation(pair.negative, request) @ direction.unit_direction)
        pos_row = {
            "experiment": cfg["name"],
            "contrast_id": pair.contrast.contrast_id,
            "example_id": pair.positive.id,
            "label": 1,
            "score": pos_score,
        }
        neg_row = {
            "experiment": cfg["name"],
            "contrast_id": pair.contrast.contrast_id,
            "example_id": pair.negative.id,
            "label": 0,
            "score": neg_score,
        }
        rows.extend([pos_row, neg_row])
        logger.log_metric(pos_row)
        logger.log_metric(neg_row)
        scores.extend([pos_score, neg_score])
        labels.extend([1, 0])
    positive_scores = [score for score, label in zip(scores, labels, strict=True) if label == 1]
    negative_scores = [score for score, label in zip(scores, labels, strict=True) if label == 0]
    summary = {
        "experiment": cfg["name"],
        "backend": backend.name,
        "behavior": cfg["behavior"],
        "rows": len(rows),
        "auroc": auroc(scores, labels),
        "mean_positive_score": float(np.mean(positive_scores)),
        "mean_negative_score": float(np.mean(negative_scores)),
        "score_gap": float(np.mean(positive_scores) - np.mean(negative_scores)),
    }
    logger.write_json("summary.json", summary)
    write_experiment_report(
        logger.run_dir / "report.md", "E002 Activation Monitor", summary, rows[:50]
    )
    write_static_dashboard(logger.run_dir.parent)
    return logger.run_dir


def run_e003(repo_root: Path, config_path: Path, backend_kind: str | None = None) -> Path:
    cfg = load_yaml(config_path)
    backend_name = backend_kind or str(cfg.get("backend", "fake"))
    model_config_path = resolve_path(str(cfg["model"]), repo_root)
    model_cfg = load_yaml(model_config_path)
    backend = build_backend(backend_name, model_config_path)
    store = _load_benchmark(repo_root, str(cfg["dataset"]))
    logger = RunLogger(
        resolve_path(str(cfg.get("output_dir", "runs")), repo_root),
        str(cfg["name"]),
        {"config": cfg, "backend": backend.name, "benchmark": store.validate()},
    )
    rows: list[dict[str, Any]] = []
    for behavior in cfg["behaviors"]:
        pairs = store.pairs(
            behavior=str(behavior),
            origin_bucket=str(cfg["origin"]),
            limit=int(cfg.get("limit_per_behavior", 16)),
        )
        for layer in cfg["layers"]:
            request = ActivationRequest(
                layer=int(layer), activation_view=str(cfg["activation_view"])
            )
            sae = None
            if backend_name == "qwen":
                from steering_research.models.qwen_scope import QwenScopeSae

                sae = QwenScopeSae(
                    repo_id=str(model_cfg["sae_repo_id"]),
                    layer=int(layer),
                    top_k=int(model_cfg.get("sae_top_k", 50)),
                    local_files_only=bool(model_cfg.get("local_files_only", False)),
                )
            deltas = rank_sae_deltas(
                backend,
                pairs,
                request,
                top_features=int(cfg.get("top_features", 25)),
                sae=sae,
            )
            for rank, result in enumerate(deltas, start=1):
                row = {
                    "experiment": cfg["name"],
                    "behavior": behavior,
                    "origin": cfg["origin"],
                    "layer": int(layer),
                    "rank": rank,
                    "feature_index": result.feature_index,
                    "delta": result.delta,
                    "positive_mean": result.positive_mean,
                    "negative_mean": result.negative_mean,
                }
                rows.append(row)
                logger.log_metric(row)
    summary = {"experiment": cfg["name"], "backend": backend.name, "rows": len(rows)}
    logger.write_json("summary.json", summary)
    write_experiment_report(logger.run_dir / "report.md", "E003 SAE Delta", summary, rows[:50])
    write_static_dashboard(logger.run_dir.parent)
    return logger.run_dir


def run_e004(repo_root: Path, config_path: Path, backend_kind: str | None = None) -> Path:
    cfg = load_yaml(config_path)
    backend_name = backend_kind or str(cfg.get("backend", "fake"))
    backend = build_backend(backend_name, resolve_path(str(cfg["model"]), repo_root))
    store = _load_benchmark(repo_root, str(cfg["dataset"]))
    logger = RunLogger(
        resolve_path(str(cfg.get("output_dir", "runs")), repo_root),
        str(cfg["name"]),
        {"config": cfg, "backend": backend.name, "benchmark": store.validate()},
    )
    pairs = store.pairs(
        behavior=str(cfg["behavior"]),
        origin_bucket=str(cfg["origin"]),
        limit=int(cfg.get("train_limit", 16)),
    )
    request = ActivationRequest(
        layer=int(cfg["layer"]), activation_view=str(cfg["activation_view"])
    )
    direction = build_direction(backend, pairs, request)
    examples = store.examples_for_bucket(
        str(cfg["eval_bucket"]),
        behavior=str(cfg["behavior"]),
        limit=int(cfg.get("eval_limit", 8)),
    )
    rows = run_steering_sweep(
        backend,
        examples,
        direction,
        [float(alpha) for alpha in cfg["alphas"]],
        max_new_tokens=int(cfg.get("max_new_tokens", 96)),
        generation_batch_size=int(cfg.get("generation_batch_size", 1)),
        on_row=logger.log_metric,
    )
    by_alpha: dict[float, list[dict[str, object]]] = {}
    for row in rows:
        by_alpha.setdefault(_as_float(row["alpha"]), []).append(row)
    aggregate_rows = []
    for alpha, alpha_rows in sorted(by_alpha.items()):
        aggregate_rows.append(
            {
                "alpha": alpha,
                "n": len(alpha_rows),
                "mean_refusal_marker": float(
                    np.mean([_as_float(r["refusal_marker"]) for r in alpha_rows])
                ),
                "mean_agreement_marker": float(
                    np.mean([_as_float(r["agreement_marker"]) for r in alpha_rows])
                ),
                "mean_length_tokens": float(
                    np.mean([_as_float(r["length_tokens"]) for r in alpha_rows])
                ),
            }
        )
    with (logger.run_dir / "tables" / "generations.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["empty"])
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "experiment": cfg["name"],
        "backend": backend.name,
        "behavior": cfg["behavior"],
        "rows": len(rows),
        "alphas": cfg["alphas"],
    }
    logger.write_json("summary.json", summary)
    logger.write_json("aggregate.json", {"rows": aggregate_rows})
    write_experiment_report(
        logger.run_dir / "report.md", "E004 Steering Eval", summary, aggregate_rows
    )
    write_static_dashboard(logger.run_dir.parent)
    return logger.run_dir


def run_e005(repo_root: Path, config_path: Path, backend_kind: str | None = None) -> Path:
    cfg = load_yaml(config_path)
    backend_name = backend_kind or str(cfg.get("backend", "fake"))
    model_config_path = resolve_path(str(cfg["model"]), repo_root)
    model_cfg = load_yaml(model_config_path)
    backend = build_backend(backend_name, model_config_path)
    store = _load_benchmark(repo_root, str(cfg["dataset"]))
    logger = RunLogger(
        resolve_path(str(cfg.get("output_dir", "runs")), repo_root),
        str(cfg["name"]),
        {"config": cfg, "backend": backend.name, "benchmark": store.validate()},
    )
    pairs = store.pairs(
        behavior=str(cfg["behavior"]),
        origin_bucket=str(cfg["origin"]),
        limit=int(cfg.get("train_limit", 16)),
    )
    request = ActivationRequest(
        layer=int(cfg["layer"]), activation_view=str(cfg["activation_view"])
    )
    sae = None
    if backend_name == "qwen":
        from steering_research.models.qwen_scope import QwenScopeSae

        sae = QwenScopeSae(
            repo_id=str(model_cfg["sae_repo_id"]),
            layer=int(cfg["layer"]),
            top_k=int(model_cfg.get("sae_top_k", 50)),
            local_files_only=bool(model_cfg.get("local_files_only", False)),
        )
    deltas = rank_sae_deltas(
        backend,
        pairs,
        request,
        top_features=int(cfg.get("top_features", 10)),
        sae=sae,
    )
    if sae is None:
        direction = build_direction(backend, pairs, request)
        steering_vector = direction.unit_direction
        selected_feature = -1
    else:
        selected_feature = deltas[0].feature_index
        steering_vector = np.asarray(sae.decoder_vector_numpy(selected_feature), dtype=np.float64)
        steering_vector = steering_vector / max(float(np.linalg.norm(steering_vector)), 1e-8)
        direction = build_direction(backend, pairs, request)
    examples = store.examples_for_bucket(
        str(cfg["eval_bucket"]),
        behavior=str(cfg["behavior"]),
        limit=int(cfg.get("eval_limit", 8)),
    )
    rows: list[dict[str, Any]] = []
    batch_size = max(1, int(cfg.get("generation_batch_size", 1)))
    generate_batch = getattr(backend, "generate_batch", None)
    for alpha in [float(x) for x in cfg["alphas"]]:
        steering = (int(cfg["layer"]), steering_vector, alpha)
        for start in range(0, len(examples), batch_size):
            batch = examples[start : start + batch_size]
            if callable(generate_batch) and batch_size > 1:
                generations = generate_batch(
                    batch,
                    max_new_tokens=int(cfg.get("max_new_tokens", 96)),
                    steering=steering,
                )
            else:
                generations = [
                    backend.generate(
                        example,
                        max_new_tokens=int(cfg.get("max_new_tokens", 96)),
                        steering=steering,
                    )
                    for example in batch
                ]
            for example, generation in zip(batch, generations, strict=True):
                scores = score_generation(generation.text)
                row = {
                    "experiment": cfg["name"],
                    "example_id": example.id,
                    "alpha": alpha,
                    "selected_feature": selected_feature,
                    "text": generation.text,
                    **scores,
                }
                rows.append(row)
                logger.log_metric(row)
    by_alpha: dict[float, list[dict[str, object]]] = {}
    for row in rows:
        by_alpha.setdefault(_as_float(row["alpha"]), []).append(row)
    aggregate_rows = []
    for alpha, alpha_rows in sorted(by_alpha.items()):
        aggregate_rows.append(
            {
                "alpha": alpha,
                "n": len(alpha_rows),
                "mean_refusal_marker": float(
                    np.mean([_as_float(r["refusal_marker"]) for r in alpha_rows])
                ),
                "mean_agreement_marker": float(
                    np.mean([_as_float(r["agreement_marker"]) for r in alpha_rows])
                ),
                "mean_length_tokens": float(
                    np.mean([_as_float(r["length_tokens"]) for r in alpha_rows])
                ),
            }
        )
    with (logger.run_dir / "tables" / "generations.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["empty"])
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "experiment": cfg["name"],
        "backend": backend.name,
        "behavior": cfg["behavior"],
        "selected_feature": selected_feature,
        "selected_feature_delta": deltas[0].delta if deltas else float("nan"),
        "fallback_caa_norm": float(np.linalg.norm(direction.direction)),
        "rows": len(rows),
    }
    logger.write_json("summary.json", summary)
    logger.write_json(
        "top_sae_features.json",
        {"rows": [result.__dict__ for result in deltas]},
    )
    logger.write_json("aggregate.json", {"rows": aggregate_rows})
    write_experiment_report(
        logger.run_dir / "report.md", "E005 SAE Feature Steering", summary, aggregate_rows
    )
    write_static_dashboard(logger.run_dir.parent)
    return logger.run_dir


def run_e007(repo_root: Path, config_path: Path, backend_kind: str | None = None) -> Path:
    cfg = load_yaml(config_path)
    backend_name = backend_kind or str(cfg.get("backend", "qwen"))
    backend = build_backend(backend_name, resolve_path(str(cfg["model"]), repo_root))
    store = _load_benchmark(repo_root, str(cfg["dataset"]))
    logger = RunLogger(
        resolve_path(str(cfg.get("output_dir", "runs")), repo_root),
        str(cfg["name"]),
        {"config": cfg, "backend": backend.name, "benchmark": store.validate()},
    )
    rows: list[dict[str, Any]] = []
    for entry in cfg["entries"]:
        behavior = str(entry["behavior"])
        origin = str(entry["origin"])
        layer = int(entry["layer"])
        activation_view = str(entry["activation_view"])
        pairs = store.pairs(
            behavior=behavior,
            origin_bucket=origin,
            limit=int(entry.get("train_limit", cfg.get("train_limit", 1_000_000))),
        )
        request = ActivationRequest(layer=layer, activation_view=activation_view)
        direction = build_direction(backend, pairs, request)
        examples = store.examples_for_bucket(
            str(entry.get("eval_bucket", cfg["eval_bucket"])),
            behavior=behavior,
            limit=int(entry.get("eval_limit", cfg.get("eval_limit", 1_000_000))),
        )
        rows.extend(
            _generate_vector_sweep(
                backend,
                examples,
                layer,
                direction.unit_direction,
                [float(x) for x in cfg["alphas"]],
                int(cfg.get("max_new_tokens", 96)),
                int(cfg.get("generation_batch_size", 1)),
                logger,
                {
                    "experiment": cfg["name"],
                    "entry": str(entry.get("name", behavior)),
                    "steering_method": "best_layer_caa",
                    "source_behavior": behavior,
                    "source_origin": origin,
                    "layer": layer,
                    "activation_view": activation_view,
                },
            )
        )
    aggregate_rows = _aggregate_marker_rows(
        rows, ["entry", "source_behavior", "layer", "activation_view", "alpha"]
    )
    summary = {
        "experiment": cfg["name"],
        "backend": backend.name,
        "rows": len(rows),
        "entries": len(cfg["entries"]),
        "alphas": cfg["alphas"],
    }
    _write_generation_outputs(logger, rows, aggregate_rows, summary, "E007 Best-Layer CAA Sweep")
    return logger.run_dir


def run_e008(repo_root: Path, config_path: Path, backend_kind: str | None = None) -> Path:
    cfg = load_yaml(config_path)
    backend_name = backend_kind or str(cfg.get("backend", "qwen"))
    backend = build_backend(backend_name, resolve_path(str(cfg["model"]), repo_root))
    store = _load_benchmark(repo_root, str(cfg["dataset"]))
    logger = RunLogger(
        resolve_path(str(cfg.get("output_dir", "runs")), repo_root),
        str(cfg["name"]),
        {"config": cfg, "backend": backend.name, "benchmark": store.validate()},
    )
    rows: list[dict[str, Any]] = []
    for source in cfg["sources"]:
        source_behavior = str(source["behavior"])
        source_origin = str(source["origin"])
        layer = int(source["layer"])
        activation_view = str(source["activation_view"])
        train_pairs = store.pairs(
            behavior=source_behavior,
            origin_bucket=source_origin,
            limit=int(source.get("train_limit", cfg.get("train_limit", 1_000_000))),
        )
        direction = build_direction(
            backend,
            train_pairs,
            ActivationRequest(layer=layer, activation_view=activation_view),
        )
        for target_behavior in cfg["target_behaviors"]:
            for target_origin in cfg["target_origins"]:
                eval_pairs = store.pairs(
                    behavior=str(target_behavior),
                    origin_bucket=str(target_origin),
                    limit=int(cfg.get("eval_limit", 1_000_000)),
                )
                if not eval_pairs:
                    continue
                metrics = evaluate_direction(backend, eval_pairs, direction)
                row = {
                    "experiment": cfg["name"],
                    "source_behavior": source_behavior,
                    "source_origin": source_origin,
                    "target_behavior": target_behavior,
                    "target_origin": target_origin,
                    "layer": layer,
                    "activation_view": activation_view,
                    **metrics,
                }
                rows.append(row)
                logger.log_metric(row)
    summary = {"experiment": cfg["name"], "backend": backend.name, "rows": len(rows)}
    logger.write_json("summary.json", summary)
    write_experiment_report(
        logger.run_dir / "report.md", "E008 Steering Specificity Matrix", summary, rows[:100]
    )
    write_static_dashboard(logger.run_dir.parent)
    return logger.run_dir


def run_e009(repo_root: Path, config_path: Path, backend_kind: str | None = None) -> Path:
    cfg = load_yaml(config_path)
    backend_name = backend_kind or str(cfg.get("backend", "qwen"))
    backend = build_backend(backend_name, resolve_path(str(cfg["model"]), repo_root))
    store = _load_benchmark(repo_root, str(cfg["dataset"]))
    logger = RunLogger(
        resolve_path(str(cfg.get("output_dir", "runs")), repo_root),
        str(cfg["name"]),
        {"config": cfg, "backend": backend.name, "benchmark": store.validate()},
    )
    behavior = str(cfg["behavior"])
    origin = str(cfg["origin"])
    layer = int(cfg["layer"])
    activation_view = str(cfg["activation_view"])
    pairs = store.pairs(behavior=behavior, origin_bucket=origin, limit=int(cfg["train_limit"]))
    request = ActivationRequest(layer=layer, activation_view=activation_view)
    base = build_direction(backend, pairs, request)
    rng = np.random.default_rng(int(cfg.get("seed", 17)))
    random_vector = rng.normal(0.0, 1.0, base.direction.shape)
    shuffled_diffs = []
    for pair in pairs:
        sign = 1.0 if rng.random() >= 0.5 else -1.0
        pos = backend.activation(pair.positive, request)
        neg = backend.activation(pair.negative, request)
        shuffled_diffs.append(sign * (pos - neg))
    shuffled = _direction_from_vector(base, np.stack(shuffled_diffs, axis=0).mean(axis=0))
    variants: list[tuple[str, np.ndarray]] = [
        ("base", base.unit_direction),
        ("opposite_sign", -base.unit_direction),
        ("random_norm_matched", unit_vector(random_vector)),
        ("shuffled_labels", shuffled.unit_direction),
    ]
    unrelated_behavior = str(cfg.get("unrelated_behavior", "hallucination"))
    unrelated_pairs = store.pairs(
        behavior=unrelated_behavior,
        origin_bucket=str(cfg.get("unrelated_origin", origin)),
        limit=int(cfg["train_limit"]),
    )
    if unrelated_pairs:
        unrelated = build_direction(backend, unrelated_pairs, request)
        variants.append((f"unrelated_{unrelated_behavior}", unrelated.unit_direction))
    synthetic_pairs = store.pairs(
        behavior=behavior,
        origin_bucket=str(cfg.get("synthetic_origin", "synthetic_contrasts")),
        limit=int(cfg["train_limit"]),
    )
    if synthetic_pairs:
        synthetic = build_direction(backend, synthetic_pairs, request)
        variants.append(("same_behavior_synthetic", synthetic.unit_direction))
    examples = store.examples_for_bucket(
        str(cfg["eval_bucket"]), behavior=behavior, limit=int(cfg.get("eval_limit", 1_000_000))
    )
    rows: list[dict[str, Any]] = []
    for variant, vector in variants:
        rows.extend(
            _generate_vector_sweep(
                backend,
                examples,
                layer,
                vector,
                [float(x) for x in cfg["alphas"]],
                int(cfg.get("max_new_tokens", 96)),
                int(cfg.get("generation_batch_size", 1)),
                logger,
                {
                    "experiment": cfg["name"],
                    "variant": variant,
                    "source_behavior": behavior,
                    "layer": layer,
                    "activation_view": activation_view,
                },
            )
        )
    aggregate_rows = _aggregate_marker_rows(rows, ["variant", "alpha"])
    summary = {
        "experiment": cfg["name"],
        "backend": backend.name,
        "behavior": behavior,
        "rows": len(rows),
        "variants": [name for name, _ in variants],
    }
    _write_generation_outputs(logger, rows, aggregate_rows, summary, "E009 Causal Controls")
    return logger.run_dir


def run_e010(repo_root: Path, config_path: Path, backend_kind: str | None = None) -> Path:
    cfg = load_yaml(config_path)
    backend_name = backend_kind or str(cfg.get("backend", "qwen"))
    model_config_path = resolve_path(str(cfg["model"]), repo_root)
    model_cfg = load_yaml(model_config_path)
    backend = build_backend(backend_name, model_config_path)
    store = _load_benchmark(repo_root, str(cfg["dataset"]))
    logger = RunLogger(
        resolve_path(str(cfg.get("output_dir", "runs")), repo_root),
        str(cfg["name"]),
        {"config": cfg, "backend": backend.name, "benchmark": store.validate()},
    )
    if backend_name != "qwen":
        msg = "E010 requires qwen backend and Qwen-Scope SAE."
        raise ValueError(msg)
    from steering_research.models.qwen_scope import QwenScopeSae

    behavior = str(cfg["behavior"])
    layer = int(cfg["layer"])
    pairs = store.pairs(
        behavior=behavior, origin_bucket=str(cfg["origin"]), limit=int(cfg["train_limit"])
    )
    request = ActivationRequest(layer=layer, activation_view=str(cfg["activation_view"]))
    sae = QwenScopeSae(
        repo_id=str(model_cfg["sae_repo_id"]),
        layer=layer,
        top_k=int(model_cfg.get("sae_top_k", 50)),
        local_files_only=bool(model_cfg.get("local_files_only", False)),
    )
    deltas = rank_sae_deltas(
        backend, pairs, request, top_features=int(cfg.get("top_features", 5)), sae=sae
    )
    examples = store.examples_for_bucket(
        str(cfg["eval_bucket"]), behavior=behavior, limit=int(cfg.get("eval_limit", 1_000_000))
    )
    rows: list[dict[str, Any]] = []
    for rank, delta in enumerate(deltas, start=1):
        vector = np.asarray(sae.decoder_vector_numpy(delta.feature_index), dtype=np.float64)
        rows.extend(
            _generate_vector_sweep(
                backend,
                examples,
                layer,
                vector,
                [float(x) for x in cfg["alphas"]],
                int(cfg.get("max_new_tokens", 96)),
                int(cfg.get("generation_batch_size", 1)),
                logger,
                {
                    "experiment": cfg["name"],
                    "feature_rank": rank,
                    "feature_index": delta.feature_index,
                    "feature_delta": delta.delta,
                    "source_behavior": behavior,
                },
            )
        )
    aggregate_rows = _aggregate_marker_rows(rows, ["feature_rank", "feature_index", "alpha"])
    summary = {
        "experiment": cfg["name"],
        "backend": backend.name,
        "behavior": behavior,
        "rows": len(rows),
        "features": [delta.__dict__ for delta in deltas],
    }
    logger.write_json("top_sae_features.json", {"rows": [delta.__dict__ for delta in deltas]})
    _write_generation_outputs(logger, rows, aggregate_rows, summary, "E010 SAE Feature Sweep")
    return logger.run_dir


def run_e011(repo_root: Path, config_path: Path, backend_kind: str | None = None) -> Path:
    cfg = load_yaml(config_path)
    backend_name = backend_kind or str(cfg.get("backend", "qwen"))
    backend = build_backend(backend_name, resolve_path(str(cfg["model"]), repo_root))
    store = _load_benchmark(repo_root, str(cfg["dataset"]))
    logger = RunLogger(
        resolve_path(str(cfg.get("output_dir", "runs")), repo_root),
        str(cfg["name"]),
        {"config": cfg, "backend": backend.name, "benchmark": store.validate()},
    )
    behavior = str(cfg["behavior"])
    origin = str(cfg["origin"])
    layer = int(cfg["layer"])
    activation_view = str(cfg["activation_view"])
    request = ActivationRequest(layer=layer, activation_view=activation_view)
    base_pairs = store.pairs(behavior=behavior, origin_bucket=origin, limit=int(cfg["train_limit"]))
    base = build_direction(backend, base_pairs, request)
    controls = []
    for control_behavior in cfg["control_behaviors"]:
        pairs = store.pairs(
            behavior=str(control_behavior), origin_bucket=origin, limit=int(cfg["train_limit"])
        )
        if pairs:
            controls.append(build_direction(backend, pairs, request).direction)
    orthogonal = _direction_from_vector(base, _orthogonalize(base.direction, controls))
    variants = [("raw", base.unit_direction), ("orthogonalized", orthogonal.unit_direction)]
    examples = store.examples_for_bucket(
        str(cfg["eval_bucket"]), behavior=behavior, limit=int(cfg.get("eval_limit", 1_000_000))
    )
    rows: list[dict[str, Any]] = []
    for variant, vector in variants:
        rows.extend(
            _generate_vector_sweep(
                backend,
                examples,
                layer,
                vector,
                [float(x) for x in cfg["alphas"]],
                int(cfg.get("max_new_tokens", 96)),
                int(cfg.get("generation_batch_size", 1)),
                logger,
                {
                    "experiment": cfg["name"],
                    "variant": variant,
                    "source_behavior": behavior,
                    "n_controls": len(controls),
                },
            )
        )
    aggregate_rows = _aggregate_marker_rows(rows, ["variant", "alpha"])
    summary = {
        "experiment": cfg["name"],
        "backend": backend.name,
        "behavior": behavior,
        "rows": len(rows),
        "control_behaviors": cfg["control_behaviors"],
        "n_controls_used": len(controls),
    }
    _write_generation_outputs(logger, rows, aggregate_rows, summary, "E011 Orthogonalized Steering")
    return logger.run_dir


def run_e012(repo_root: Path, config_path: Path, backend_kind: str | None = None) -> Path:
    cfg = load_yaml(config_path)
    backend_name = backend_kind or str(cfg.get("backend", "qwen"))
    backend = build_backend(backend_name, resolve_path(str(cfg["model"]), repo_root))
    store = _load_benchmark(repo_root, str(cfg["dataset"]))
    logger = RunLogger(
        resolve_path(str(cfg.get("output_dir", "runs")), repo_root),
        str(cfg["name"]),
        {"config": cfg, "backend": backend.name, "benchmark": store.validate()},
    )
    rows: list[dict[str, Any]] = []
    for behavior in cfg["behaviors"]:
        for train_origin in cfg["train_origins"]:
            train_pairs = store.pairs(
                behavior=str(behavior),
                origin_bucket=str(train_origin),
                limit=int(cfg.get("train_limit", 1_000_000)),
            )
            if not train_pairs:
                continue
            request = ActivationRequest(
                layer=int(cfg["layer"]), activation_view=str(cfg["activation_view"])
            )
            direction = build_direction(backend, train_pairs, request)
            for eval_origin in cfg["eval_origins"]:
                eval_pairs = store.pairs(
                    behavior=str(behavior),
                    origin_bucket=str(eval_origin),
                    limit=int(cfg.get("eval_limit", 1_000_000)),
                )
                if not eval_pairs:
                    continue
                metrics = evaluate_direction(backend, eval_pairs, direction)
                row = {
                    "experiment": cfg["name"],
                    "behavior": behavior,
                    "train_origin": train_origin,
                    "eval_origin": eval_origin,
                    "layer": int(cfg["layer"]),
                    "activation_view": cfg["activation_view"],
                    **metrics,
                }
                rows.append(row)
                logger.log_metric(row)
    summary = {"experiment": cfg["name"], "backend": backend.name, "rows": len(rows)}
    logger.write_json("summary.json", summary)
    write_experiment_report(logger.run_dir / "report.md", "E012 Origin Transfer", summary, rows)
    write_static_dashboard(logger.run_dir.parent)
    return logger.run_dir


def run_e013(repo_root: Path, config_path: Path, backend_kind: str | None = None) -> Path:
    cfg = load_yaml(config_path)
    backend_name = backend_kind or str(cfg.get("backend", "qwen"))
    backend = build_backend(backend_name, resolve_path(str(cfg["model"]), repo_root))
    store = _load_benchmark(repo_root, str(cfg["dataset"]))
    logger = RunLogger(
        resolve_path(str(cfg.get("output_dir", "runs")), repo_root),
        str(cfg["name"]),
        {"config": cfg, "backend": backend.name, "benchmark": store.validate()},
    )
    behavior = str(cfg["behavior"])
    layer = int(cfg["layer"])
    request = ActivationRequest(layer=layer, activation_view=str(cfg["activation_view"]))
    pairs = store.pairs(
        behavior=behavior, origin_bucket=str(cfg["origin"]), limit=int(cfg["train_limit"])
    )
    direction = build_direction(backend, pairs, request)
    scores = []
    for pair in pairs:
        scores.append(float(backend.activation(pair.positive, request) @ direction.unit_direction))
        scores.append(float(backend.activation(pair.negative, request) @ direction.unit_direction))
    threshold = float(np.quantile(scores, float(cfg.get("threshold_quantile", 0.75))))
    examples = store.examples_for_bucket(
        str(cfg["eval_bucket"]), behavior=behavior, limit=int(cfg.get("eval_limit", 1_000_000))
    )
    rows: list[dict[str, Any]] = []
    rows.extend(
        _generate_vector_sweep(
            backend,
            examples,
            layer,
            direction.unit_direction,
            [float(x) for x in cfg["alphas"]],
            int(cfg.get("max_new_tokens", 96)),
            int(cfg.get("generation_batch_size", 1)),
            logger,
            {"experiment": cfg["name"], "variant": "always", "threshold": threshold},
        )
    )
    batch_size = max(1, int(cfg.get("generation_batch_size", 1)))
    generate_batch = getattr(backend, "generate_batch", None)
    for alpha in [float(x) for x in cfg["alphas"]]:
        steered: list[Any] = []
        unsteered: list[Any] = []
        scores_by_id = {}
        for example in examples:
            score = float(backend.activation(example, request) @ direction.unit_direction)
            scores_by_id[example.id] = score
            if score > threshold:
                steered.append(example)
            else:
                unsteered.append(example)
        for use_steering, batch_source in ((True, steered), (False, unsteered)):
            for start in range(0, len(batch_source), batch_size):
                batch = batch_source[start : start + batch_size]
                if not batch:
                    continue
                steering = (layer, direction.unit_direction, alpha) if use_steering else None
                if callable(generate_batch) and batch_size > 1:
                    generations = generate_batch(
                        batch, max_new_tokens=int(cfg.get("max_new_tokens", 96)), steering=steering
                    )
                else:
                    generations = [
                        backend.generate(
                            example,
                            max_new_tokens=int(cfg.get("max_new_tokens", 96)),
                            steering=steering,
                        )
                        for example in batch
                    ]
                for example, generation in zip(batch, generations, strict=True):
                    row = {
                        "experiment": cfg["name"],
                        "variant": "dynamic",
                        "alpha": alpha,
                        "example_id": example.id,
                        "monitor_score": scores_by_id[example.id],
                        "threshold": threshold,
                        "applied_steering": float(use_steering),
                        "text": generation.text,
                        **score_generation(generation.text),
                    }
                    rows.append(row)
                    logger.log_metric(row)
    aggregate_rows = _aggregate_marker_rows(rows, ["variant", "alpha"])
    for row in aggregate_rows:
        subset = [
            item
            for item in rows
            if item["variant"] == row["variant"] and item["alpha"] == row["alpha"]
        ]
        if subset and "applied_steering" in subset[0]:
            row["mean_applied_steering"] = float(
                np.mean([_as_float(item.get("applied_steering", 1.0)) for item in subset])
            )
    summary = {
        "experiment": cfg["name"],
        "backend": backend.name,
        "behavior": behavior,
        "rows": len(rows),
        "threshold": threshold,
    }
    _write_generation_outputs(logger, rows, aggregate_rows, summary, "E013 Dynamic Steering")
    return logger.run_dir


def run_e014(repo_root: Path, config_path: Path, backend_kind: str | None = None) -> Path:
    cfg = load_yaml(config_path)
    backend_name = backend_kind or str(cfg.get("backend", "qwen"))
    backend = build_backend(backend_name, resolve_path(str(cfg["model"]), repo_root))
    store = _load_benchmark(repo_root, str(cfg["dataset"]))
    logger = RunLogger(
        resolve_path(str(cfg.get("output_dir", "runs")), repo_root),
        str(cfg["name"]),
        {"config": cfg, "backend": backend.name, "benchmark": store.validate()},
    )
    behavior = str(cfg["behavior"])
    origin = str(cfg["origin"])
    activation_view = str(cfg["activation_view"])
    layer_vectors = {}
    for layer in cfg["layers"]:
        request = ActivationRequest(layer=int(layer), activation_view=activation_view)
        pairs = store.pairs(behavior=behavior, origin_bucket=origin, limit=int(cfg["train_limit"]))
        layer_vectors[int(layer)] = build_direction(backend, pairs, request).unit_direction
    steering_groups = [
        {
            "name": group["name"],
            "divide_alpha": bool(group.get("divide_alpha", True)),
            "hooks": [
                {"layer": int(layer), "vector": layer_vectors[int(layer)]}
                for layer in group["layers"]
            ],
        }
        for group in cfg["groups"]
    ]
    examples = store.examples_for_bucket(
        str(cfg["eval_bucket"]), behavior=behavior, limit=int(cfg.get("eval_limit", 1_000_000))
    )
    rows = _generate_multi_steering_sweep(
        backend,
        examples,
        steering_groups,
        [float(x) for x in cfg["alphas"]],
        int(cfg.get("max_new_tokens", 96)),
        int(cfg.get("generation_batch_size", 1)),
        logger,
    )
    aggregate_rows = _aggregate_marker_rows(rows, ["variant", "n_hooks", "alpha"])
    summary = {
        "experiment": cfg["name"],
        "backend": backend.name,
        "behavior": behavior,
        "rows": len(rows),
        "groups": cfg["groups"],
    }
    _write_generation_outputs(logger, rows, aggregate_rows, summary, "E014 Multi-Layer Steering")
    return logger.run_dir


def run_e015(repo_root: Path, config_path: Path, backend_kind: str | None = None) -> Path:
    cfg = load_yaml(config_path)
    backend_name = backend_kind or str(cfg.get("backend", "qwen"))
    backend = build_backend(backend_name, resolve_path(str(cfg["model"]), repo_root))
    store = _load_benchmark(repo_root, str(cfg["dataset"]))
    logger = RunLogger(
        resolve_path(str(cfg.get("output_dir", "runs")), repo_root),
        str(cfg["name"]),
        {"config": cfg, "backend": backend.name, "benchmark": store.validate()},
    )
    behavior = str(cfg["behavior"])
    origin = str(cfg["origin"])
    activation_view = str(cfg["activation_view"])
    rows: list[dict[str, Any]] = []
    for source_layer in cfg["source_layers"]:
        source_request = ActivationRequest(layer=int(source_layer), activation_view=activation_view)
        train_pairs = store.pairs(
            behavior=behavior, origin_bucket=origin, limit=int(cfg["train_limit"])
        )
        source_direction = build_direction(backend, train_pairs, source_request)
        for target_layer in cfg["target_layers"]:
            target_direction = DirectionResult(
                behavior=source_direction.behavior,
                layer=int(target_layer),
                activation_view=activation_view,
                component=source_direction.component,
                n_pairs=source_direction.n_pairs,
                direction=source_direction.direction,
                unit_direction=source_direction.unit_direction,
            )
            eval_pairs = store.pairs(
                behavior=behavior, origin_bucket=origin, limit=int(cfg.get("eval_limit", 1_000_000))
            )
            metrics = evaluate_direction(backend, eval_pairs, target_direction)
            row = {
                "experiment": cfg["name"],
                "behavior": behavior,
                "origin": origin,
                "source_layer": int(source_layer),
                "target_layer": int(target_layer),
                "activation_view": activation_view,
                **metrics,
            }
            rows.append(row)
            logger.log_metric(row)
    best = max(rows, key=lambda row: float(row["direction_accuracy"])) if rows else {}
    summary = {
        "experiment": cfg["name"],
        "backend": backend.name,
        "behavior": behavior,
        "rows": len(rows),
        "best_source_layer": best.get("source_layer", ""),
        "best_target_layer": best.get("target_layer", ""),
        "best_direction_accuracy": best.get("direction_accuracy", float("nan")),
    }
    logger.write_json("summary.json", summary)
    write_experiment_report(
        logger.run_dir / "report.md", "E015 Layer-Fraction Transfer", summary, rows
    )
    write_static_dashboard(logger.run_dir.parent)
    return logger.run_dir


def run_qwen_limited_smoke(repo_root: Path, limit: int = 2) -> Path:
    cfg = load_yaml(repo_root / "configs" / "experiments" / "e005_sae_feature_steering.yaml")
    model_config_path = repo_root / "configs" / "models" / "qwen35_2b.yaml"
    model_cfg = load_yaml(model_config_path)
    backend = build_backend("qwen", model_config_path)
    store = _load_benchmark(repo_root, str(cfg["dataset"]))
    logger = RunLogger(
        resolve_path(str(cfg.get("output_dir", "runs")), repo_root),
        "qwen_limited_smoke",
        {"config": cfg, "backend": backend.name, "benchmark": store.validate(), "limit": limit},
    )
    pairs = store.pairs(
        behavior=str(cfg["behavior"]),
        origin_bucket=str(cfg["origin"]),
        limit=max(2, limit),
    )
    request = ActivationRequest(layer=0, activation_view=str(cfg["activation_view"]))
    direction = build_direction(backend, pairs, request)
    metrics = evaluate_direction(backend, pairs, direction)
    logger.log_metric({"stage": "caa_forward", **metrics})

    from steering_research.models.qwen_scope import QwenScopeSae

    sae = QwenScopeSae(
        repo_id=str(model_cfg["sae_repo_id"]),
        layer=0,
        top_k=int(model_cfg.get("sae_top_k", 50)),
        local_files_only=bool(model_cfg.get("local_files_only", False)),
    )
    deltas = rank_sae_deltas(backend, pairs, request, top_features=3, sae=sae)
    for rank, result in enumerate(deltas, start=1):
        logger.log_metric(
            {
                "stage": "qwen_scope_sae_delta",
                "rank": rank,
                "feature_index": result.feature_index,
                "delta": result.delta,
            }
        )

    examples = store.examples_for_bucket(
        str(cfg["eval_bucket"]),
        behavior=str(cfg["behavior"]),
        limit=1,
    )
    selected_feature = deltas[0].feature_index
    steering_vector = np.asarray(sae.decoder_vector_numpy(selected_feature), dtype=np.float64)
    steering_vector = steering_vector / max(float(np.linalg.norm(steering_vector)), 1e-8)
    generations = []
    for alpha in [0.0, -0.5]:
        generation = backend.generate(
            examples[0],
            max_new_tokens=16,
            steering=(0, steering_vector, alpha),
        )
        row = {
            "stage": "steered_generation",
            "alpha": alpha,
            "selected_feature": selected_feature,
            "text": generation.text,
        }
        generations.append(row)
        logger.log_metric(row)
    summary = {
        "experiment": "qwen_limited_smoke",
        "backend": backend.name,
        "pairs": len(pairs),
        "selected_feature": selected_feature,
        "direction_accuracy": metrics["direction_accuracy"],
        "generations": len(generations),
    }
    logger.write_json("summary.json", summary)
    logger.write_json("top_sae_features.json", {"rows": [result.__dict__ for result in deltas]})
    write_experiment_report(
        logger.run_dir / "report.md",
        "Qwen Limited Smoke",
        summary,
        generations,
    )
    write_static_dashboard(logger.run_dir.parent)
    return logger.run_dir


def write_smoke_summary(
    repo_root: Path, run_dirs: list[Path], runs_root: Path | None = None
) -> Path:
    out_root = runs_root or repo_root / "runs"
    out = out_root / "smoke_summary.json"
    payload = {"run_dirs": [str(path) for path in run_dirs]}
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out


def run_real_smoke_suite(repo_root: Path) -> list[Path]:
    smoke_root = repo_root / "configs" / "experiments" / "smoke"
    run_dirs = [
        run_e001(repo_root, smoke_root / "qwen2b_e001.yaml", "qwen"),
        run_e002(repo_root, smoke_root / "qwen2b_e002.yaml", "qwen"),
        run_e003(repo_root, smoke_root / "qwen2b_e003.yaml", "qwen"),
        run_e004(repo_root, smoke_root / "qwen2b_e004.yaml", "qwen"),
        run_e005(repo_root, smoke_root / "qwen2b_e005.yaml", "qwen"),
    ]
    from steering_research.training import run_lora_sft

    run_dirs.append(run_lora_sft(repo_root, smoke_root / "qwen2b_e006.yaml"))
    runs_root = run_dirs[0].parent
    write_smoke_summary(repo_root, run_dirs, runs_root=runs_root)
    write_static_dashboard(runs_root)
    return run_dirs
