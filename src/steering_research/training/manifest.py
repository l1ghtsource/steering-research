from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingPlan:
    """Typed manifest for future LoRA/SFT/DPO and feature-penalty runs."""

    name: str
    base_model: str
    dataset: str
    method: str
    output_dir: str
    seed: int = 17
