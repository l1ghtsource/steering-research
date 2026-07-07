from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, TypeVar

import numpy as np

from steering_research.cli.backend import build_backend
from steering_research.data import BenchmarkStore
from steering_research.methods import build_direction, evaluate_direction, rank_sae_deltas
from steering_research.metrics.basic import auroc
from steering_research.models.base import ActivationRequest
from steering_research.reports.dashboard import write_static_dashboard
from steering_research.reports.markdown import write_experiment_report
from steering_research.runtime import RunLogger, load_yaml, resolve_path
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
    )
    for row in rows:
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
    rows = []
    for example in examples:
        for alpha in [float(x) for x in cfg["alphas"]]:
            generation = backend.generate(
                example,
                max_new_tokens=int(cfg.get("max_new_tokens", 96)),
                steering=(int(cfg["layer"]), steering_vector, alpha),
            )
            row = {
                "experiment": cfg["name"],
                "example_id": example.id,
                "alpha": alpha,
                "selected_feature": selected_feature,
                "text": generation.text,
            }
            rows.append(row)
            logger.log_metric(row)
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
    write_experiment_report(
        logger.run_dir / "report.md", "E005 SAE Feature Steering", summary, rows[:20]
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
