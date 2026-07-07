from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REQUIRED_RUN_FILES = [
    "manifest.json",
    "metrics.jsonl",
    "summary.json",
    "report.md",
    "run.log",
]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        msg = f"JSON root must be object: {path}"
        raise TypeError(msg)
    return data


def verify_run_dir(run_dir: Path) -> dict[str, Any]:
    missing = [name for name in REQUIRED_RUN_FILES if not (run_dir / name).exists()]
    empty = [
        name
        for name in REQUIRED_RUN_FILES
        if (run_dir / name).exists() and (run_dir / name).stat().st_size == 0
    ]
    summary = _load_json(run_dir / "summary.json") if not missing else {}
    metrics_lines = 0
    metrics_path = run_dir / "metrics.jsonl"
    if metrics_path.exists():
        metrics_lines = sum(
            1 for line in metrics_path.read_text(encoding="utf-8").splitlines() if line
        )
    return {
        "run_dir": str(run_dir),
        "ok": not missing and not empty and metrics_lines > 0,
        "missing": missing,
        "empty": empty,
        "metrics_lines": metrics_lines,
        "experiment": summary.get("experiment", run_dir.name),
    }


def verify_runs(runs_root: Path) -> dict[str, Any]:
    run_dirs = sorted(path for path in runs_root.iterdir() if path.is_dir())
    rows = [verify_run_dir(path) for path in run_dirs]
    dashboard = runs_root / "dashboard.html"
    ok = bool(rows) and all(bool(row["ok"]) for row in rows) and dashboard.exists()
    return {
        "ok": ok,
        "runs_root": str(runs_root),
        "dashboard": str(dashboard),
        "dashboard_exists": dashboard.exists(),
        "run_count": len(rows),
        "rows": rows,
    }
