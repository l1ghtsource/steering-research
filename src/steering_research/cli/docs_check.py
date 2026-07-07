from __future__ import annotations

from pathlib import Path

from steering_research.runtime.config import repo_root_from_cwd


def main() -> int:
    repo_root = repo_root_from_cwd()
    required = [
        repo_root / "docs" / "index.md",
        repo_root / "docs" / "methods.md",
        repo_root / "docs" / "datasets.md",
        repo_root / "docs" / "experiments.md",
        repo_root / "docs" / "reports.md",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        for path in missing:
            print(f"missing docs file: {path}")
        return 1
    for path in required:
        text = path.read_text(encoding="utf-8")
        if not text.strip().startswith("#"):
            print(f"docs file must start with a heading: {path}")
            return 1
    for path in Path(repo_root / "docs").glob("*.md"):
        if "\t" in path.read_text(encoding="utf-8"):
            print(f"tabs are not allowed in docs: {path}")
            return 1
    print("docs check passed")
    return 0
