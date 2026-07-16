#!/usr/bin/env python3
"""Build a balanced Label Studio import file from rendered flat-IDS samples.

The task file deliberately contains one rendered instance for each selected
Unicode character.  It supplies automatic IDS/component hints, but annotators
must verify the hints and draw the actual component rectangles.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from urllib.parse import quote

from ocr_ids.dataset import SampleRecord, read_jsonl
from ocr_ids.ids import parse_ids
from ocr_ids.structure_scope import flat_structure_kind


STRUCTURE_ORDER = (
    "left_right",
    "top_bottom",
    "left_middle_right",
    "top_middle_bottom",
)
# Preserve scarce three-component structures while keeping the initial project
# dominated by the structures most likely to appear in ordinary text.
DEFAULT_QUOTAS = {
    "left_right": 600,
    "top_bottom": 300,
    "left_middle_right": 50,
    "top_middle_bottom": 50,
}


def choose_records(records: list[SampleRecord], count: int) -> list[SampleRecord]:
    by_structure: dict[str, list[SampleRecord]] = defaultdict(list)
    seen_characters: set[str] = set()
    for record in sorted(records, key=lambda item: (item.character or "", item.sample_id)):
        kind = flat_structure_kind(record.ids)
        if not kind or not record.character or not record.image_path:
            continue
        # Same character in many fonts does not create a new first-round task.
        if record.character in seen_characters:
            continue
        seen_characters.add(record.character)
        by_structure[kind].append(record)

    scale = count / sum(DEFAULT_QUOTAS.values())
    quotas = {kind: round(DEFAULT_QUOTAS[kind] * scale) for kind in STRUCTURE_ORDER}
    quotas["left_right"] += count - sum(quotas.values())
    selected: list[SampleRecord] = []
    remaining: list[SampleRecord] = []
    for kind in STRUCTURE_ORDER:
        bucket = by_structure[kind]
        selected.extend(bucket[: quotas[kind]])
        remaining.extend(bucket[quotas[kind] :])

    # If a category cannot fill its target, fill the project with other valid
    # glyphs rather than silently exporting fewer tasks.
    if len(selected) < count:
        selected.extend(sorted(remaining, key=lambda item: item.sample_id)[: count - len(selected)])
    return selected[:count]


def task_from_record(record: SampleRecord, local_files_root: Path) -> dict[str, object]:
    assert record.image_path and record.character
    image_path = Path(record.image_path).resolve()
    relative = image_path.relative_to(local_files_root.resolve()).as_posix()
    root = parse_ids(record.ids)
    components = [child.to_prefix() for child in root.children]
    hints = {f"component_{index}_hint": component for index, component in enumerate(components, 1)}
    for index in range(len(components) + 1, 4):
        hints[f"component_{index}_hint"] = ""
    return {
        "data": {
            "image": f"/data/local-files/?d={quote(relative)}",
            "sample_id": record.sample_id,
            "character_hint": record.character,
            "ids_hint": record.ids,
            "structure_hint": flat_structure_kind(record.ids),
            **hints,
        }
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--local-files-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--count", type=int, default=1000)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")

    records = choose_records(list(read_jsonl(args.manifest)), args.count)
    if len(records) < args.count:
        raise SystemExit(f"Only {len(records)} valid, distinct flat glyphs are available; need {args.count}.")
    tasks = [task_from_record(record, args.local_files_root) for record in records]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(tasks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    distribution = Counter(task["data"]["structure_hint"] for task in tasks)
    print(json.dumps({"output": str(args.output), "tasks": len(tasks), "distribution": distribution}, ensure_ascii=False))


if __name__ == "__main__":
    main()
