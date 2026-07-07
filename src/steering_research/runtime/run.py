from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return tolist()
    msg = f"Object is not JSON serializable: {type(value)!r}"
    raise TypeError(msg)


class RunLogger:
    def __init__(self, output_root: Path, name: str, manifest: dict[str, Any]) -> None:
        stamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
        self.run_dir = output_root / f"{stamp}_{name}"
        self.run_dir.mkdir(parents=True, exist_ok=False)
        (self.run_dir / "artifacts").mkdir()
        (self.run_dir / "tables").mkdir()
        (self.run_dir / "figures").mkdir()
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.log_path = self.run_dir / "run.log"
        self.write_json("manifest.json", manifest)
        self.log_event("run_created", {"name": name})

    def write_json(self, name: str, payload: dict[str, Any]) -> Path:
        path = self.run_dir / name
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, default=_json_default)
            handle.write("\n")
        return path

    def write_text(self, name: str, text: str) -> Path:
        path = self.run_dir / name
        path.write_text(text, encoding="utf-8")
        return path

    def log_metric(self, row: dict[str, Any]) -> None:
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, default=_json_default))
            handle.write("\n")
        self.log_event("metric", row)

    def log_event(self, event: str, payload: dict[str, Any] | None = None) -> None:
        row = {
            "time": datetime.now(tz=UTC).isoformat(),
            "event": event,
            "payload": payload or {},
        }
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, default=_json_default))
            handle.write("\n")
