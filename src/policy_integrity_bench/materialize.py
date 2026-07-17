#!/usr/bin/env python3
"""Materialize a secret-seed Track E evaluation view and a separate scoring key."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from policy_integrity_bench.core import content_sha256, rerender_track_e_item


def load_items(data_dir: Path, split: str) -> list[dict[str, Any]]:
    path = data_dir / split / "track_e_items.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")


def materialize(
    data_dir: Path, input_dir: Path, key_dir: Path, split: str, seed: int
) -> dict[str, Any]:
    input_dir = input_dir.resolve()
    key_dir = key_dir.resolve()
    if input_dir == key_dir or input_dir in key_dir.parents or key_dir in input_dir.parents:
        raise ValueError("input and scoring-key directories must be separate and non-nested")
    rendered_items = [rerender_track_e_item(item, seed) for item in load_items(data_dir, split)]
    random.Random(f"pib-order:{seed}:{split}").shuffle(rendered_items)
    inputs = []
    scoring_key = []
    for item in rendered_items:
        eval_id = hashlib.sha256(f"{seed}:{item['item_id']}".encode()).hexdigest()[:24]
        inputs.append(
            {
                "eval_id": eval_id,
                "prompt": item["prompt"],
                "prompt_hash": item["prompt_hash"],
            }
        )
        scoring_key.append(
            {
                "eval_id": eval_id,
                "family_id": item["family_id"],
                "split": item["split"],
                "domain": item["domain"],
                "generator_group": item["generator_group"],
                "solver_kind": item["world"]["solver_kind"],
                "condition": item["condition"],
                "target": item["target"],
                "prompt_hash": item["prompt_hash"],
            }
        )

    scoring_key.sort(key=lambda record: record["eval_id"])
    input_dir.mkdir(parents=True, exist_ok=False)
    key_dir.mkdir(parents=True, exist_ok=False)
    input_path = input_dir / "track_e_inputs.jsonl"
    key_path = key_dir / "track_e_scoring_key.jsonl"
    write_jsonl(input_path, inputs)
    write_jsonl(key_path, scoring_key)
    manifest = {
        "schema_version": "1.0",
        "split": split,
        "items": len(inputs),
        "families": len({item["family_id"] for item in scoring_key}),
        "render_seed_sha256": hashlib.sha256(str(seed).encode()).hexdigest(),
        "inputs_hash": content_sha256(input_path.read_text()),
        "scoring_key_hash": content_sha256(key_path.read_text()),
        "warning": (
            "Keep the seed and scoring key outside model and method-selection contexts. "
            "Inference must be frozen before access and stateless across eval_ids; do not "
            "fit, retrieve, normalize, or otherwise adapt from the evaluation corpus."
        ),
    }
    (input_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    key_manifest = {
        **manifest,
        "warning": "Do not expose this directory to inference or method-selection workers.",
    }
    (key_dir / "manifest.json").write_text(
        json.dumps(key_manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--input-output", type=Path, required=True)
    parser.add_argument("--key-output", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "validation"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args(argv)
    manifest = materialize(
        args.data.resolve(),
        args.input_output.resolve(),
        args.key_output.resolve(),
        args.split,
        args.seed,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
