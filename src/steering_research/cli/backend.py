from __future__ import annotations

from pathlib import Path
from typing import Any

from steering_research.models import ActivationBackend, FakeActivationBackend
from steering_research.runtime.config import load_yaml


def build_backend(kind: str, model_config_path: Path | None = None) -> ActivationBackend:
    if kind == "fake":
        return FakeActivationBackend()
    if kind == "qwen":
        if model_config_path is None:
            msg = "Qwen backend requires a model config path"
            raise ValueError(msg)
        from steering_research.models.qwen import QwenActivationBackend

        cfg: dict[str, Any] = load_yaml(model_config_path)
        return QwenActivationBackend(
            model_id=str(cfg["model_id"]),
            device=str(cfg.get("device", "auto")),
            dtype=str(cfg.get("dtype", "auto")),
            name=str(cfg.get("name", "qwen")),
        )
    msg = f"Unknown backend: {kind}"
    raise ValueError(msg)
