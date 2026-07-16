#!/usr/bin/env python3
"""Keep only non-nested left/right and top/bottom IDS structures for phase one."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path

from ocr_ids.dataset import read_jsonl, write_jsonl
from ocr_ids.structure_scope import flat_structure_kind


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", help="Canonical IDS JSONL")
    parser.add_argument("--output", required=True, help="In-scope JSONL")
    parser.add_argument("--out-of-scope-output", required=True, help="Excluded JSONL")
    parser.add_argument("--report", required=True)
    return parser.parse_args()


def main() -> int:
    args = arguments()
    accepted = []
    excluded = []
    by_structure: Counter[str] = Counter()
    for record in read_jsonl(args.manifest):
        kind = flat_structure_kind(record.ids)
        if kind is None:
            excluded.append(
                replace(
                    record,
                    metadata={
                        **record.metadata,
                        "scope_version": "flat-structure-v1",
                        "review_reasons": ["out_of_scope_structure"],
                    },
                )
            )
            continue
        by_structure[kind] += 1
        accepted.append(
            replace(
                record,
                metadata={
                    **record.metadata,
                    "scope_version": "flat-structure-v1",
                    "structure_kind": kind,
                },
            )
        )
    write_jsonl(accepted, args.output)
    write_jsonl(excluded, args.out_of_scope_output)
    report = {
        "scope_version": "flat-structure-v1",
        "accepted_records": len(accepted),
        "out_of_scope_records": len(excluded),
        "accepted_by_structure": dict(sorted(by_structure.items())),
        "policy": "Only flat ⿰, ⿱, ⿲, ⿳ IDS trees; no enclosure or nested structure.",
    }
    destination = Path(args.report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
