from __future__ import annotations

import numpy as np
import pytest

from steering_research.models.qwen_scope import QwenScopeSae


def test_qwen_scope_sae_loads_local_directory(tmp_path) -> None:  # type: ignore[no-untyped-def]
    torch = pytest.importorskip("torch")
    state = {
        "W_enc": torch.eye(2),
        "W_dec": torch.eye(2),
        "b_enc": torch.zeros(2),
        "b_dec": torch.zeros(2),
    }
    torch.save(state, tmp_path / "layer3.sae.pt")

    sae = QwenScopeSae(
        repo_id=str(tmp_path),
        layer=3,
        top_k=1,
        local_files_only=True,
    )
    acts = sae.encode_numpy(np.array([1.0, 2.0]))
    decoder_vector = sae.decoder_vector_numpy(1)

    assert acts.shape == (2,)
    assert acts[0] == 0.0
    assert acts[1] == 2.0
    assert decoder_vector.tolist() == [0.0, 1.0]
