from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from steering_research.data.schema import Example
from steering_research.models.base import (
    ActivationRequest,
    FloatArray,
    GenerationResult,
    SequenceLogprobResult,
)


def _seed_from_text(text: str) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


@dataclass
class FakeActivationBackend:
    """Deterministic local backend for testing the full pipeline without model downloads."""

    hidden_size: int = 128
    name: str = "fake"

    def activation(self, example: Example, request: ActivationRequest) -> FloatArray:
        seed = _seed_from_text(
            f"{example.id}|{example.text}|{request.layer}|{request.activation_view}|{request.component}"
        )
        rng = np.random.default_rng(seed)
        vector = rng.normal(0.0, 0.25, self.hidden_size)
        for axis in example.behavior_axes:
            axis_seed = _seed_from_text(axis)
            axis_rng = np.random.default_rng(axis_seed)
            axis_vector = axis_rng.normal(0.0, 1.0, self.hidden_size)
            axis_vector = axis_vector / np.linalg.norm(axis_vector)
            label_value = example.labels.get(axis)
            if label_value == 1:
                vector += axis_vector
            elif label_value == 0:
                vector -= axis_vector
        return vector.astype(np.float64)

    def generate(
        self,
        example: Example,
        max_new_tokens: int = 96,
        steering: tuple[int, FloatArray, float] | None = None,
    ) -> GenerationResult:
        steer_suffix = ""
        if steering is not None:
            layer, _, alpha = steering
            steer_suffix = f" [steered layer={layer} alpha={alpha:.3f}]"
        behavior = example.behavior_axes[0] if example.behavior_axes else "capability"
        text = f"Fake {behavior} response for {example.id}.{steer_suffix}"
        return GenerationResult(
            text=text[: max_new_tokens * 8],
            prompt=example.prompt_text,
            metadata={"backend": self.name, "example_id": example.id},
        )

    def sequence_logprob(
        self,
        prompt: str,
        completion: str,
        steering: tuple[int, FloatArray, float] | None = None,
        position_mode: str = "all",
    ) -> SequenceLogprobResult:
        token_count = max(1, len(completion.split()))
        seed = _seed_from_text(f"{prompt}|{completion}|{position_mode}")
        base = -0.1 * token_count - (seed % 997) / 9970.0
        steering_shift = 0.0
        if steering is not None:
            _layer, direction, alpha = steering
            checksum = float(direction[: min(8, direction.shape[0])].sum())
            steering_shift = 0.01 * float(alpha) * checksum
        logprob = base + steering_shift
        return SequenceLogprobResult(
            prompt=prompt,
            completion=completion,
            logprob=logprob,
            mean_logprob=logprob / token_count,
            token_count=token_count,
            metadata={
                "backend": self.name,
                "steering": steering is not None,
                "position_mode": position_mode,
            },
        )

    def sequence_logprob_batch(
        self,
        prompts: list[str],
        completions: list[str],
        steering: tuple[int, FloatArray, float] | None = None,
        position_mode: str = "all",
    ) -> list[SequenceLogprobResult]:
        return [
            self.sequence_logprob(
                prompt,
                completion,
                steering=steering,
                position_mode=position_mode,
            )
            for prompt, completion in zip(prompts, completions, strict=True)
        ]
