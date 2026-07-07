from __future__ import annotations

from collections.abc import Callable, Iterable

from steering_research.data.schema import Example
from steering_research.methods.caa import DirectionResult
from steering_research.models.base import ActivationBackend
from steering_research.scoring import score_generation


def _chunks(items: list[Example], size: int) -> Iterable[list[Example]]:
    for index in range(0, len(items), size):
        yield items[index : index + size]


def run_steering_sweep(
    backend: ActivationBackend,
    examples: list[Example],
    direction: DirectionResult,
    alphas: list[float],
    max_new_tokens: int = 96,
    generation_batch_size: int = 1,
    on_row: Callable[[dict[str, object]], None] | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    batch_size = max(1, generation_batch_size)
    generate_batch = getattr(backend, "generate_batch", None)
    for alpha in alphas:
        steering = (direction.layer, direction.unit_direction, alpha)
        for batch in _chunks(examples, batch_size):
            if callable(generate_batch) and batch_size > 1:
                generations = generate_batch(
                    batch,
                    max_new_tokens=max_new_tokens,
                    steering=steering,
                )
            else:
                generations = [
                    backend.generate(
                        example,
                        max_new_tokens=max_new_tokens,
                        steering=steering,
                    )
                    for example in batch
                ]
            for example, generation in zip(batch, generations, strict=True):
                scores = score_generation(generation.text)
                row: dict[str, object] = {
                    "example_id": example.id,
                    "behavior": ",".join(example.behavior_axes),
                    "alpha": alpha,
                    "text": generation.text,
                    **scores,
                }
                rows.append(row)
                if on_row is not None:
                    on_row(row)
    return rows
