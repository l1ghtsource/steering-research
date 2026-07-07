from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from steering_research.data.schema import Contrast, ContrastPair, Example


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


class BenchmarkStore:
    def __init__(
        self,
        root: Path,
        examples_path: str = "processed/examples.jsonl",
        contrasts_path: str = "processed/contrasts.jsonl",
        clean_splits_path: str = "processed/eval_splits_clean.json",
    ) -> None:
        self.root = root
        self.examples_path = root / examples_path
        self.contrasts_path = root / contrasts_path
        self.clean_splits_path = root / clean_splits_path
        self.examples = {
            row["id"]: Example.from_json(row) for row in load_jsonl(self.examples_path)
        }
        self.contrasts = {
            row["contrast_id"]: Contrast.from_json(row) for row in load_jsonl(self.contrasts_path)
        }
        self.clean_splits = load_json(self.clean_splits_path)

    @classmethod
    def from_repo_root(cls, repo_root: Path) -> BenchmarkStore:
        return cls(repo_root / "external" / "LatentBehaviorBench")

    def validate(self) -> dict[str, int]:
        missing_positive = 0
        missing_negative = 0
        for contrast in self.contrasts.values():
            missing_positive += sum(
                item_id not in self.examples for item_id in contrast.positive_item_ids
            )
            missing_negative += sum(
                item_id not in self.examples for item_id in contrast.negative_item_ids
            )
        return {
            "examples": len(self.examples),
            "contrasts": len(self.contrasts),
            "missing_positive_refs": missing_positive,
            "missing_negative_refs": missing_negative,
        }

    def contrast_ids_for_origin(self, origin_bucket: str) -> list[str]:
        extraction = self.clean_splits.get("extraction", {})
        ids = extraction.get(origin_bucket, [])
        return [str(item_id) for item_id in ids]

    def eval_ids_for_bucket(self, bucket: str) -> list[str]:
        for section in ("eval", "controls"):
            values = self.clean_splits.get(section, {})
            if bucket in values:
                return [str(item_id) for item_id in values[bucket]]
        msg = f"Unknown clean split bucket: {bucket}"
        raise KeyError(msg)

    def pairs(
        self,
        behavior: str | None = None,
        origin_bucket: str | None = None,
        limit: int | None = None,
    ) -> list[ContrastPair]:
        if origin_bucket is None:
            candidates = list(self.contrasts.values())
        else:
            candidates = [
                self.contrasts[item_id] for item_id in self.contrast_ids_for_origin(origin_bucket)
            ]
        pairs: list[ContrastPair] = []
        for contrast in candidates:
            if behavior is not None and contrast.behavior != behavior:
                continue
            for pos_id, neg_id in zip(
                contrast.positive_item_ids, contrast.negative_item_ids, strict=False
            ):
                pairs.append(
                    ContrastPair(
                        contrast=contrast,
                        positive=self.examples[pos_id],
                        negative=self.examples[neg_id],
                    )
                )
                if limit is not None and len(pairs) >= limit:
                    return pairs
        return pairs

    def examples_for_bucket(
        self,
        bucket: str,
        behavior: str | None = None,
        limit: int | None = None,
    ) -> list[Example]:
        rows: list[Example] = []
        for item_id in self.eval_ids_for_bucket(bucket):
            example = self.examples[item_id]
            if behavior is not None and behavior not in example.behavior_axes:
                continue
            rows.append(example)
            if limit is not None and len(rows) >= limit:
                return rows
        return rows

    def contrast_counts_by_behavior(self) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for contrast in self.contrasts.values():
            counts[contrast.behavior] += 1
        return dict(sorted(counts.items()))
