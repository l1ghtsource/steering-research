from __future__ import annotations

from steering_research.data.schema import Example
from steering_research.methods.caa import DirectionResult
from steering_research.models.base import ActivationBackend
from steering_research.scoring import score_generation


def run_steering_sweep(
    backend: ActivationBackend,
    examples: list[Example],
    direction: DirectionResult,
    alphas: list[float],
    max_new_tokens: int = 96,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for example in examples:
        for alpha in alphas:
            generation = backend.generate(
                example,
                max_new_tokens=max_new_tokens,
                steering=(direction.layer, direction.unit_direction, alpha),
            )
            scores = score_generation(generation.text)
            rows.append(
                {
                    "example_id": example.id,
                    "behavior": ",".join(example.behavior_axes),
                    "alpha": alpha,
                    "text": generation.text,
                    **scores,
                }
            )
    return rows
