from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from steering_research.data.schema import Example
from steering_research.models.base import (
    ActivationRequest,
    FloatArray,
    GenerationResult,
    SequenceLogprobResult,
)


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

    def _position_delta(
        self,
        hidden: Any,
        addition: Any,
        prompt_lengths: list[int] | None,
        position_mode: str,
    ) -> Any:
        torch = self._torch
        if position_mode == "all":
            return addition.view(1, 1, -1).expand_as(hidden)
        if prompt_lengths is None:
            msg = f"Position mode {position_mode!r} requires prompt lengths."
            raise ValueError(msg)
        delta = torch.zeros_like(hidden)
        seq_len = int(hidden.shape[1])
        for batch_index, prompt_len in enumerate(prompt_lengths):
            boundary = max(0, min(int(prompt_len), seq_len))
            if position_mode == "prompt":
                if boundary > 0:
                    delta[batch_index, :boundary, :] = addition
            elif position_mode == "answer":
                if boundary < seq_len:
                    delta[batch_index, boundary:, :] = addition
            elif position_mode == "last_prompt":
                index = max(0, min(boundary - 1, seq_len - 1))
                delta[batch_index, index, :] = addition
            elif position_mode == "first_answer":
                if boundary < seq_len:
                    delta[batch_index, boundary, :] = addition
            else:
                msg = f"Unsupported steering position mode: {position_mode}"
                raise ValueError(msg)
        return delta

    def _install_steering_hooks(
        self,
        steerings: list[tuple[int, FloatArray, float]],
        position_mode: str = "all",
        prompt_lengths: list[int] | None = None,
    ) -> list[Any]:
        hooks = []
        if not steerings:
            return hooks
        torch = self._torch
        for layer, direction, alpha in steerings:
            direction_tensor = torch.tensor(direction, device=self.device, dtype=self.model.dtype)
            direction_tensor = direction_tensor / direction_tensor.norm().clamp_min(1e-8)
            addition = alpha * direction_tensor

            def _hook(
                _module: Any,
                _inputs: Any,
                output: Any,
                addition: Any = addition,
                position_mode: str = position_mode,
                prompt_lengths: list[int] | None = prompt_lengths,
            ) -> Any:
                hidden = output[0] if isinstance(output, tuple) else output
                hidden = hidden + self._position_delta(
                    hidden, addition, prompt_lengths, position_mode
                )
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

    def sequence_logprob(
        self,
        prompt: str,
        completion: str,
        steering: tuple[int, FloatArray, float] | None = None,
        position_mode: str = "all",
    ) -> SequenceLogprobResult:
        return self.sequence_logprob_batch(
            [prompt],
            [completion],
            steering=steering,
            position_mode=position_mode,
        )[0]

    def sequence_logprob_batch(
        self,
        prompts: list[str],
        completions: list[str],
        steering: tuple[int, FloatArray, float] | None = None,
        position_mode: str = "all",
    ) -> list[SequenceLogprobResult]:
        if len(prompts) != len(completions):
            msg = "prompts and completions must have the same length"
            raise ValueError(msg)
        if not prompts:
            return []
        torch = self._torch
        prompt_lengths = [
            int(
                self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)[
                    "input_ids"
                ].shape[-1]
            )
            for prompt in prompts
        ]
        full_texts = [
            f"{prompt}{completion}" for prompt, completion in zip(prompts, completions, strict=True)
        ]
        old_padding_side = str(self.tokenizer.padding_side)
        self.tokenizer.padding_side = "right"
        try:
            full_inputs = self.tokenizer(
                full_texts,
                return_tensors="pt",
                add_special_tokens=True,
                padding=True,
            )
        finally:
            self.tokenizer.padding_side = old_padding_side
        input_ids = full_inputs["input_ids"].to(self.device)
        attention_mask = full_inputs["attention_mask"].to(self.device)
        sequence_lengths = [int(value) for value in attention_mask.sum(dim=1).tolist()]
        hooks = self._install_steering_hooks(
            [] if steering is None else [steering],
            position_mode=position_mode,
            prompt_lengths=prompt_lengths,
        )
        try:
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
                logits = outputs.logits[:, :-1, :]
                labels = input_ids[:, 1:]
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                token_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
                results = []
                for index, (prompt, completion) in enumerate(
                    zip(prompts, completions, strict=True)
                ):
                    seq_len = sequence_lengths[index]
                    prompt_len = prompt_lengths[index]
                    start = max(0, min(prompt_len - 1, seq_len - 1))
                    end = max(start, seq_len - 1)
                    completion_log_probs = token_log_probs[index, start:end]
                    token_count = int(completion_log_probs.numel())
                    logprob = (
                        float(completion_log_probs.sum().detach().float().cpu())
                        if token_count
                        else float("nan")
                    )
                    mean_logprob = logprob / token_count if token_count else float("nan")
                    results.append(
                        SequenceLogprobResult(
                            prompt=prompt,
                            completion=completion,
                            logprob=logprob,
                            mean_logprob=mean_logprob,
                            token_count=token_count,
                            metadata={
                                "backend": self.name,
                                "model_id": self.model_id,
                                "steering": steering is not None,
                                "position_mode": position_mode,
                                "prompt_tokens": prompt_len,
                                "sequence_tokens": seq_len,
                            },
                        )
                    )
        finally:
            for hook in hooks:
                hook.remove()
        return results

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
