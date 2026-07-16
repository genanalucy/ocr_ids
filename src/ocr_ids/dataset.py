"""Versionable JSONL records for image-to-IDS experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Iterable, Iterator, Literal

from .ids import parse_ids


LabelStatus = Literal["automatic", "reviewed", "gold"]


@dataclass(slots=True)
class SampleRecord:
    sample_id: str
    ids: str
    image_path: str | None = None
    character: str | None = None
    codepoint: str | None = None
    glyph_region: str | None = None
    source: str | None = None
    source_version: str | None = None
    label_status: LabelStatus = "automatic"
    alternatives: list[str] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.sample_id:
            raise ValueError("sample_id 不能为空")
        parse_ids(self.ids)
        for alternative in self.alternatives:
            parse_ids(alternative)
        if self.character and len(self.character) != 1:
            raise ValueError("character 必须恰好包含一个 Unicode 字符")
        if self.label_status not in {"automatic", "reviewed", "gold"}:
            raise ValueError(f"未知 label_status: {self.label_status}")

    def to_json(self) -> str:
        self.validate()
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "SampleRecord":
        record = cls(**value)  # type: ignore[arg-type]
        record.validate()
        return record


def read_jsonl(path: str | Path) -> Iterator[SampleRecord]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError("记录不是 JSON 对象")
                yield SampleRecord.from_dict(value)
            except Exception as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc


def write_jsonl(records: Iterable[SampleRecord], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.to_json() + "\n")

