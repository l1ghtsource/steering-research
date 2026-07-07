from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Message:
    role: str
    content: str


@dataclass(frozen=True)
class Example:
    id: str
    source: str
    behavior_axes: tuple[str, ...]
    task_family: str
    format: str
    language: str
    messages: tuple[Message, ...]
    reference: dict[str, Any]
    labels: dict[str, Any]
    scoring: dict[str, Any]
    split: str
    metadata: dict[str, Any]

    @classmethod
    def from_json(cls, row: dict[str, Any]) -> Example:
        messages = tuple(Message(str(m["role"]), str(m["content"])) for m in row["messages"])
        return cls(
            id=str(row["id"]),
            source=str(row["source"]),
            behavior_axes=tuple(str(x) for x in row.get("behavior_axes", [])),
            task_family=str(row["task_family"]),
            format=str(row["format"]),
            language=str(row["language"]),
            messages=messages,
            reference=dict(row.get("reference", {})),
            labels=dict(row.get("labels", {})),
            scoring=dict(row.get("scoring", {})),
            split=str(row["split"]),
            metadata=dict(row.get("metadata", {})),
        )

    @property
    def text(self) -> str:
        return "\n".join(f"{message.role}: {message.content}" for message in self.messages)

    @property
    def prompt_text(self) -> str:
        prompt_messages = [m for m in self.messages if m.role != "assistant"]
        if not prompt_messages:
            prompt_messages = list(self.messages)
        return "\n".join(f"{message.role}: {message.content}" for message in prompt_messages)

    @property
    def assistant_text(self) -> str:
        return "\n".join(m.content for m in self.messages if m.role == "assistant")


@dataclass(frozen=True)
class Contrast:
    contrast_id: str
    behavior: str
    contrast_type: str
    positive_item_ids: tuple[str, ...]
    negative_item_ids: tuple[str, ...]
    matched_on: tuple[str, ...]
    activation_views: tuple[str, ...]
    recommended_methods: tuple[str, ...]
    split: str
    pair_type: str
    data_origin: str
    source: str
    notes: str

    @classmethod
    def from_json(cls, row: dict[str, Any]) -> Contrast:
        return cls(
            contrast_id=str(row["contrast_id"]),
            behavior=str(row["behavior"]),
            contrast_type=str(row["contrast_type"]),
            positive_item_ids=tuple(str(x) for x in row["positive_item_ids"]),
            negative_item_ids=tuple(str(x) for x in row["negative_item_ids"]),
            matched_on=tuple(str(x) for x in row.get("matched_on", [])),
            activation_views=tuple(str(x) for x in row.get("activation_views", [])),
            recommended_methods=tuple(str(x) for x in row.get("recommended_methods", [])),
            split=str(row["split"]),
            pair_type=str(row.get("pair_type", "")),
            data_origin=str(row.get("data_origin", "")),
            source=str(row.get("source", "")),
            notes=str(row.get("notes", "")),
        )


@dataclass(frozen=True)
class ContrastPair:
    contrast: Contrast
    positive: Example
    negative: Example
