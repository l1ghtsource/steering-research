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
