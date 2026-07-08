from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from steering_research.data.schema import Example
from steering_research.models.base import ActivationRequest, FloatArray, GenerationResult


def _torch_dtype(dtype: str, torch: Any) -> Any:
    if dtype == "float32":
        return torch.float32
    if dtype in {"float16", "fp16"}:
        return torch.float16
    if dtype in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if dtype == "auto":
        return "auto"
    msg = f"Unsupported dtype: {dtype}"
    raise ValueError(msg)


def _device(device: str, torch: Any) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class QwenActivationBackend:
    model_id: str
    device: str = "auto"
    dtype: str = "auto"
    local_files_only: bool = False
    cache_activations: bool = True
    name: str = "qwen"
    hidden_size: int = field(init=False)
    model: Any = field(init=False)
    tokenizer: Any = field(init=False)
    _torch: Any = field(init=False)
    _activation_cache: dict[tuple[str, str, int, str], FloatArray] = field(init=False)

    def __post_init__(self) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            msg = "Install model dependencies with `uv sync --extra model` to use Qwen backend."
            raise RuntimeError(msg) from exc

        self._torch = torch
        self.device = _device(self.device, torch)
        kwargs: dict[str, Any] = {"trust_remote_code": True}
        torch_dtype = _torch_dtype(self.dtype, torch)
        if torch_dtype != "auto":
            kwargs["torch_dtype"] = torch_dtype
        kwargs["local_files_only"] = self.local_files_only
        tokenizer: Any = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            local_files_only=self.local_files_only,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        self.tokenizer = tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
        model_to: Any = self.model.to
        model_to(self.device)
        self.model.eval()
        self.hidden_size = int(self.model.config.hidden_size)
        self._activation_cache = {}

    def _encode(self, example: Example) -> tuple[Any, int]:
        prompt = example.prompt_text
        full = example.text
        prompt_inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        full_inputs = self.tokenizer(full, return_tensors="pt", add_special_tokens=True)
        prompt_len = int(prompt_inputs["input_ids"].shape[-1])
        full_inputs = {key: value.to(self.device) for key, value in full_inputs.items()}
        return full_inputs, prompt_len

    def _activation_cache_key(
        self, example: Example, request: ActivationRequest
    ) -> tuple[str, str, int, str]:
        return (example.id, request.component, request.layer, request.activation_view)

    def _cache_example_activations(self, example: Example, component: str) -> None:
        torch = self._torch
        inputs, prompt_len = self._encode(example)
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True, use_cache=False)
        for layer, layer_hidden in enumerate(out.hidden_states[1:]):
            hidden = layer_hidden[0]
            seq_len = int(hidden.shape[0])
            last_prompt_index = max(0, min(prompt_len - 1, seq_len - 1))
            first_assistant_index = max(0, min(prompt_len, seq_len - 1))
            answer_start = max(0, min(prompt_len, seq_len - 1))
            vectors = {
                "last_prompt_token": hidden[last_prompt_index],
                "first_assistant_token": hidden[first_assistant_index],
                "assistant_answer_mean": hidden[answer_start:].mean(dim=0),
                "mean_answer_representation": hidden[answer_start:].mean(dim=0),
                "last_answer_token": hidden[-1],
            }
            for activation_view, vector in vectors.items():
                key = (example.id, component, layer, activation_view)
                self._activation_cache[key] = vector.detach().float().cpu().numpy()

    def activation(self, example: Example, request: ActivationRequest) -> FloatArray:
        key = self._activation_cache_key(example, request)
        if self.cache_activations:
            if key not in self._activation_cache:
                self._cache_example_activations(example, request.component)
            if key not in self._activation_cache:
                msg = f"Activation request is outside cached model layers/views: {request}"
                raise KeyError(msg)
            return self._activation_cache[key].astype(np.float64)

        self._cache_example_activations(example, request.component)
        if key not in self._activation_cache:
            msg = f"Activation request is outside model layers/views: {request}"
            raise KeyError(msg)
        vector = self._activation_cache[key].astype(np.float64)
        self._activation_cache.clear()
        return vector

    def _install_steering_hooks(self, steerings: list[tuple[int, FloatArray, float]]) -> list[Any]:
        hooks = []
        if not steerings:
            return hooks
        torch = self._torch
        for layer, direction, alpha in steerings:
            direction_tensor = torch.tensor(direction, device=self.device, dtype=self.model.dtype)
            direction_tensor = direction_tensor / direction_tensor.norm().clamp_min(1e-8)

            def _hook(
                _module: Any,
                _inputs: Any,
                output: Any,
                direction_tensor: Any = direction_tensor,
                alpha: float = alpha,
            ) -> Any:
                hidden = output[0] if isinstance(output, tuple) else output
                hidden = hidden + alpha * direction_tensor.view(1, 1, -1)
                if isinstance(output, tuple):
                    return (hidden, *output[1:])
                return hidden

            hooks.append(self.model.model.layers[layer].register_forward_hook(_hook))
        return hooks

    def _install_steering_hook(self, steering: tuple[int, FloatArray, float] | None) -> Any | None:
        hooks = self._install_steering_hooks([] if steering is None else [steering])
        return hooks[0] if hooks else None

    def generate(
        self,
        example: Example,
        max_new_tokens: int = 96,
        steering: tuple[int, FloatArray, float] | None = None,
    ) -> GenerationResult:
        torch = self._torch
        prompt = example.prompt_text
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        hook = self._install_steering_hook(steering)
        try:
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
        finally:
            if hook is not None:
                hook.remove()
        text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return GenerationResult(
            text=text,
            prompt=prompt,
            metadata={
                "backend": self.name,
                "model_id": self.model_id,
                "steering": steering is not None,
            },
        )

    def generate_batch(
        self,
        examples: list[Example],
        max_new_tokens: int = 96,
        steering: tuple[int, FloatArray, float] | None = None,
    ) -> list[GenerationResult]:
        if not examples:
            return []
        torch = self._torch
        prompts = [example.prompt_text for example in examples]
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            add_special_tokens=True,
            padding=True,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        hook = self._install_steering_hook(steering)
        try:
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
        finally:
            if hook is not None:
                hook.remove()
        return [
            GenerationResult(
                text=self.tokenizer.decode(row, skip_special_tokens=True),
                prompt=prompt,
                metadata={
                    "backend": self.name,
                    "model_id": self.model_id,
                    "steering": steering is not None,
                    "batch_size": len(examples),
                },
            )
            for row, prompt in zip(output_ids, prompts, strict=True)
        ]

    def generate_batch_multi_steering(
        self,
        examples: list[Example],
        steerings: list[tuple[int, FloatArray, float]],
        max_new_tokens: int = 96,
    ) -> list[GenerationResult]:
        if not examples:
            return []
        torch = self._torch
        prompts = [example.prompt_text for example in examples]
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            add_special_tokens=True,
            padding=True,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        hooks = self._install_steering_hooks(steerings)
        try:
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
        finally:
            for hook in hooks:
                hook.remove()
        return [
            GenerationResult(
                text=self.tokenizer.decode(row, skip_special_tokens=True),
                prompt=prompt,
                metadata={
                    "backend": self.name,
                    "model_id": self.model_id,
                    "steering": bool(steerings),
                    "batch_size": len(examples),
                    "n_steering_hooks": len(steerings),
                },
            )
            for row, prompt in zip(output_ids, prompts, strict=True)
        ]
