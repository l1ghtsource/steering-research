from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from steering_research.data.schema import Example

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ActivationRequest:
    layer: int
    activation_view: str
    component: str = "residual"


@dataclass(frozen=True)
class GenerationResult:
    text: str
    prompt: str
    metadata: dict[str, object]


@dataclass(frozen=True)
class SequenceLogprobResult:
    prompt: str
    completion: str
    logprob: float
    mean_logprob: float
    token_count: int
    metadata: dict[str, object]


class ActivationBackend(Protocol):
    name: str
    hidden_size: int

    def activation(self, example: Example, request: ActivationRequest) -> FloatArray: ...

    def generate(
        self,
        example: Example,
        max_new_tokens: int = 96,
        steering: tuple[int, FloatArray, float] | None = None,
    ) -> GenerationResult: ...

    def sequence_logprob(
        self,
        prompt: str,
        completion: str,
        steering: tuple[int, FloatArray, float] | None = None,
        position_mode: str = "all",
    ) -> SequenceLogprobResult: ...

    def sequence_logprob_batch(
        self,
        prompts: list[str],
        completions: list[str],
        steering: tuple[int, FloatArray, float] | None = None,
        position_mode: str = "all",
    ) -> list[SequenceLogprobResult]: ...
