from __future__ import annotations

from pathlib import Path
from typing import Any


def write_experiment_report(
    path: Path, title: str, summary: dict[str, Any], rows: list[dict[str, Any]]
) -> None:
    lines = [f"# {title}", ""]
    lines.append("## Summary")
    lines.append("")
    for key, value in summary.items():
        lines.append(f"- `{key}`: {value}")
    lines.append("")
    if rows:
        keys = list(rows[0])
        lines.append("## Metrics")
        lines.append("")
        lines.append("| " + " | ".join(keys) + " |")
        lines.append("| " + " | ".join("---" for _ in keys) + " |")
        for row in rows:
            values = [str(row.get(key, "")) for key in keys]
            lines.append("| " + " | ".join(values) + " |")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
