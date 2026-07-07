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
    name: str = "qwen"
    hidden_size: int = field(init=False)
    model: Any = field(init=False)
    tokenizer: Any = field(init=False)
    _torch: Any = field(init=False)

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
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            local_files_only=self.local_files_only,
        )
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
        model_to: Any = self.model.to
        model_to(self.device)
        self.model.eval()
        self.hidden_size = int(self.model.config.hidden_size)

    def _encode(self, example: Example) -> tuple[Any, int]:
        prompt = example.prompt_text
        full = example.text
        prompt_inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        full_inputs = self.tokenizer(full, return_tensors="pt", add_special_tokens=True)
        prompt_len = int(prompt_inputs["input_ids"].shape[-1])
        full_inputs = {key: value.to(self.device) for key, value in full_inputs.items()}
        return full_inputs, prompt_len

    def activation(self, example: Example, request: ActivationRequest) -> FloatArray:
        torch = self._torch
        inputs, prompt_len = self._encode(example)
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True, use_cache=False)
        hidden = out.hidden_states[request.layer + 1][0]
        seq_len = int(hidden.shape[0])
        if request.activation_view == "last_prompt_token":
            index = max(0, min(prompt_len - 1, seq_len - 1))
            vector = hidden[index]
        elif request.activation_view == "first_assistant_token":
            index = max(0, min(prompt_len, seq_len - 1))
            vector = hidden[index]
        elif request.activation_view in {"assistant_answer_mean", "mean_answer_representation"}:
            start = max(0, min(prompt_len, seq_len - 1))
            vector = hidden[start:].mean(dim=0)
        elif request.activation_view == "last_answer_token":
            vector = hidden[-1]
        else:
            index = max(0, min(prompt_len - 1, seq_len - 1))
            vector = hidden[index]
        return vector.detach().float().cpu().numpy().astype(np.float64)

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
        hook = None
        if steering is not None:
            layer, direction, alpha = steering
            direction_tensor = torch.tensor(direction, device=self.device, dtype=self.model.dtype)
            direction_tensor = direction_tensor / direction_tensor.norm().clamp_min(1e-8)

            def _hook(_module: Any, _inputs: Any, output: Any) -> Any:
                hidden = output[0] if isinstance(output, tuple) else output
                hidden = hidden + alpha * direction_tensor.view(1, 1, -1)
                if isinstance(output, tuple):
                    return (hidden, *output[1:])
                return hidden

            hook = self.model.model.layers[layer].register_forward_hook(_hook)
        try:
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=False
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
