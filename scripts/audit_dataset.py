#!/usr/bin/env python3
"""Produce a compact, reproducible IDS dataset audit report."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from ocr_ids.dataset import read_jsonl
from ocr_ids.ids import parse_ids, validate_ids


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-k", type=int, default=50)
    return parser.parse_args()


def main() -> int:
    args = arguments()
    records = list(read_jsonl(args.manifest))
    sample_ids: Counter[str] = Counter()
    characters: set[str] = set()
    ids_values: set[str] = set()
    character_ids: dict[str, set[str]] = defaultdict(set)
    leaf_counts: Counter[str] = Counter()
    operator_counts: Counter[str] = Counter()
    depth_counts: Counter[int] = Counter()
    region_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    strict_problem_counts: Counter[str] = Counter()
    entity_records = 0
    unknown_records = 0
    alternative_records = 0

    for record in records:
        sample_ids[record.sample_id] += 1
        if record.character:
            characters.add(record.character)
            character_ids[record.character].add(record.ids)
        ids_values.add(record.ids)
        root = parse_ids(record.ids)
        leaf_counts.update(root.leaves())
        operator_counts.update(root.operators())
        depth_counts[root.depth] += 1
        region_counts[record.glyph_region or "unspecified"] += 1
        label_counts[record.label_status] += 1
        strict_problems = validate_ids(record.ids, strict_terminals=True)
        if strict_problems:
            strict_problem_counts["records"] += 1
            strict_problem_counts.update(strict_problems)
        if any(leaf.startswith("&") for leaf in root.leaves()):
            entity_records += 1
        if "？" in root.leaves():
            unknown_records += 1
        if record.alternatives:
            alternative_records += 1

    report = {
        "manifest": str(Path(args.manifest).resolve()),
        "records": len(records),
        "unique_sample_ids": len(sample_ids),
        "duplicate_sample_ids": sum(count > 1 for count in sample_ids.values()),
        "unique_characters": len(characters),
        "unique_ids": len(ids_values),
        "unique_leaves": len(leaf_counts),
        "characters_with_multiple_primary_ids": sum(
            len(values) > 1 for values in character_ids.values()
        ),
        "records_with_alternatives": alternative_records,
        "records_with_entities": entity_records,
        "records_with_unknown_leaf": unknown_records,
        "records_failing_strict_terminal_check": strict_problem_counts.pop("records", 0),
        "depth_distribution": dict(sorted(depth_counts.items())),
        "operator_counts": operator_counts.most_common(),
        "region_counts": region_counts.most_common(),
        "label_counts": label_counts.most_common(),
        "top_leaves": leaf_counts.most_common(args.top_k),
        "top_strict_terminal_problems": strict_problem_counts.most_common(args.top_k),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in list(report)[:12]}, ensure_ascii=False, indent=2))
    print(f"report={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

