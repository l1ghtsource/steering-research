from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class QwenScopeSae:
    repo_id: str
    layer: int
    top_k: int = 50
    _state: dict[str, Any] | None = None

    def load(self) -> dict[str, Any]:
        if self._state is not None:
            return self._state
        try:
            import torch
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            msg = "Install model dependencies with `uv sync --extra model` to use Qwen-Scope SAE."
            raise RuntimeError(msg) from exc

        path = hf_hub_download(self.repo_id, filename=f"layer{self.layer}.sae.pt")
        state = torch.load(Path(path), map_location="cpu")
        required = {"W_enc", "W_dec", "b_enc", "b_dec"}
        missing = required.difference(state)
        if missing:
            msg = f"Unsupported Qwen-Scope SAE file; missing keys: {sorted(missing)}"
            raise KeyError(msg)
        self._state = state
        return state

    def encode(self, residual: Any) -> Any:
        try:
            import torch
        except ImportError as exc:
            msg = "Install model dependencies with `uv sync --extra model` to use Qwen-Scope SAE."
            raise RuntimeError(msg) from exc
        state = self.load()
        w_enc = state["W_enc"].to(residual.device, dtype=residual.dtype)
        b_enc = state["b_enc"].to(residual.device, dtype=residual.dtype)
        pre_acts = residual @ w_enc.T + b_enc
        topk_vals, topk_idx = torch.topk(pre_acts, self.top_k, dim=-1)
        acts = torch.zeros_like(pre_acts)
        acts.scatter_(-1, topk_idx, topk_vals)
        return acts

    def encode_numpy(self, residual: Any) -> Any:
        try:
            import torch
        except ImportError as exc:
            msg = "Install model dependencies with `uv sync --extra model` to use Qwen-Scope SAE."
            raise RuntimeError(msg) from exc
        tensor = torch.as_tensor(residual, dtype=torch.float32)
        return self.encode(tensor).detach().cpu().numpy()

    def decoder_vector_numpy(self, feature_index: int) -> Any:
        state = self.load()
        vector = state["W_dec"][:, feature_index]
        return vector.detach().float().cpu().numpy()
