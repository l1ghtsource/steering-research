from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


def collect_summaries(runs_root: Path) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for path in sorted(runs_root.glob("*/summary.json")):
        with path.open("r", encoding="utf-8") as handle:
            row = json.load(handle)
        row["run_dir"] = str(path.parent)
        summaries.append(row)
    return summaries


def write_static_dashboard(runs_root: Path) -> Path:
    rows = collect_summaries(runs_root)
    columns = sorted({key for row in rows for key in row})
    out = runs_root / "dashboard.html"
    style = (
        "body{font-family:system-ui;margin:24px}"
        "table{border-collapse:collapse}"
        "td,th{border:1px solid #ddd;padding:6px 8px}"
        "th{background:#f6f8fa}"
    )
    lines = [
        "<!doctype html>",
        "<meta charset='utf-8'>",
        "<title>Steering Research Dashboard</title>",
        f"<style>{style}</style>",
        "<h1>Steering Research Dashboard</h1>",
    ]
    if rows:
        lines.append("<table>")
        lines.append("<tr>" + "".join(f"<th>{html.escape(col)}</th>" for col in columns) + "</tr>")
        lines.extend(
            (
                "<tr>"
                + "".join(f"<td>{html.escape(str(row.get(col, '')))}</td>" for col in columns)
                + "</tr>"
            )
            for row in rows
        )
        lines.append("</table>")
    else:
        lines.append("<p>No run summaries found.</p>")
    out.write_text("\n".join(lines), encoding="utf-8")
    return out
