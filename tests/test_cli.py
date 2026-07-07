from __future__ import annotations

from pathlib import Path

import yaml

from steering_research.cli.experiments import run_e001, run_e002, run_e003, run_e004, run_e005


def _tmp_config(repo_root: Path, source: str, tmp_path: Path) -> Path:
    source_path = repo_root / "configs" / "experiments" / source
    with source_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    cfg["output_dir"] = str(tmp_path)
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
