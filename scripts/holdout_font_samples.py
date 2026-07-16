#!/usr/bin/env python3
"""Create a deterministic font-held-out test manifest from rendered records."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
import hashlib

from ocr_ids.dataset import SampleRecord, read_jsonl, write_jsonl


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--train-output", required=True)
    parser.add_argument("--test-output", required=True)
    return parser.parse_args()


def selected_index(key: str, count: int) -> int:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % count


def main() -> int:
    args = arguments()
    groups: dict[str, list[SampleRecord]] = defaultdict(list)
    for record in read_jsonl(args.manifest):
        groups[record.character or record.sample_id].append(record)
    train: list[SampleRecord] = []
    test: list[SampleRecord] = []
    for key, values in sorted(groups.items()):
        ordered = sorted(values, key=lambda record: record.sample_id)
        if len(ordered) < 2:
            train.extend(ordered)
            continue
        held_out = selected_index(key, len(ordered))
        for index, record in enumerate(ordered):
            split = "test_closed" if index == held_out else "train"
            transformed = replace(record, metadata={**record.metadata, "split": split})
            (test if index == held_out else train).append(transformed)
    write_jsonl(train, args.train_output)
    write_jsonl(test, args.test_output)
    print(f"train={len(train)} test_closed={len(test)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
