from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        msg = f"YAML root must be a mapping: {path}"
        raise TypeError(msg)
    return data


def resolve_path(path: str | Path, repo_root: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return repo_root / candidate


def repo_root_from_cwd(cwd: Path | None = None) -> Path:
    start = Path.cwd() if cwd is None else cwd
    for path in (start, *start.parents):
        if (path / "pyproject.toml").exists() and (path / "src" / "steering_research").exists():
            return path
    msg = "Could not locate steering-research repo root"
    raise RuntimeError(msg)
