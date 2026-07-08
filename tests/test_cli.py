from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from steering_research.cli.experiments import (
    run_e001,
    run_e002,
    run_e003,
    run_e004,
    run_e005,
    run_e016,
    run_e017,
    run_e018,
)


def _tmp_config(
    repo_root: Path,
    source: str,
    tmp_path: Path,
    overrides: dict[str, Any] | None = None,
) -> Path:
    source_path = repo_root / "configs" / "experiments" / source
    with source_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    cfg["output_dir"] = str(tmp_path)
    if overrides is not None:
        cfg.update(overrides)
    out = tmp_path / source_path.name
    with out.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)
    return out


def test_experiment_entrypoints_write_reports(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    e001 = run_e001(repo_root, _tmp_config(repo_root, "e001_mean_direction.yaml", tmp_path), "fake")
    e002 = run_e002(
        repo_root,
        _tmp_config(repo_root, "e002_activation_monitor.yaml", tmp_path),
        "fake",
    )
    e003 = run_e003(repo_root, _tmp_config(repo_root, "e003_sae_delta.yaml", tmp_path), "fake")
    e004 = run_e004(repo_root, _tmp_config(repo_root, "e004_steering_eval.yaml", tmp_path), "fake")
    e005 = run_e005(
        repo_root,
        _tmp_config(repo_root, "e005_sae_feature_steering.yaml", tmp_path),
        "fake",
    )
    for run_dir in [e001, e002, e003, e004, e005]:
        assert (run_dir / "manifest.json").exists()
        assert (run_dir / "metrics.jsonl").exists()
        assert (run_dir / "summary.json").exists()
        assert (run_dir / "report.md").exists()
        assert (run_dir / "run.log").exists()


def test_phase3_entrypoints_write_reports(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    common_overrides: dict[str, Any] = {
        "train_limit": 4,
        "eval_limit": 2,
        "pair_limit": 8,
    }
    e016_cfg = _tmp_config(
        repo_root,
        "phase3/qwen35_2b/e016_forced_choice.yaml",
        tmp_path,
        {
            **common_overrides,
            "entries": [
                {
                    "name": "sycophancy_source_l18",
                    "behavior": "sycophancy",
                    "origin": "source_backed_contrasts",
                    "layer": 18,
                    "activation_view": "assistant_answer_mean",
                }
            ],
            "alphas": [0.0, 1.0],
        },
    )
    e017_cfg = _tmp_config(
        repo_root,
        "phase3/qwen35_2b/e017_calibrated_alpha.yaml",
        tmp_path,
        {
            **common_overrides,
            "entries": [
                {
                    "name": "sycophancy_source_l18",
                    "behavior": "sycophancy",
                    "origin": "source_backed_contrasts",
                    "layer": 18,
                    "activation_view": "assistant_answer_mean",
                }
            ],
            "alpha_coefficients": [0.0, 1.0],
        },
    )
    e018_cfg = _tmp_config(
        repo_root,
        "phase3/qwen35_2b/e018_position_steering.yaml",
        tmp_path,
        {
            **common_overrides,
            "entries": [
                {
                    "name": "sycophancy_source_l18",
                    "behavior": "sycophancy",
                    "origin": "source_backed_contrasts",
                    "layer": 18,
                    "activation_view": "assistant_answer_mean",
                }
            ],
            "alphas": [0.0, 1.0],
            "position_modes": ["all", "prompt"],
        },
    )
    run_dirs = [
        run_e016(repo_root, e016_cfg, "fake"),
        run_e017(repo_root, e017_cfg, "fake"),
        run_e018(repo_root, e018_cfg, "fake"),
    ]
    for run_dir in run_dirs:
        assert (run_dir / "manifest.json").exists()
        assert (run_dir / "metrics.jsonl").exists()
        assert (run_dir / "summary.json").exists()
        assert (run_dir / "aggregate.json").exists()
        assert (run_dir / "tables" / "forced_choice.csv").exists()
        assert (run_dir / "report.md").exists()
        assert (run_dir / "run.log").exists()
