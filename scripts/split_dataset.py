#!/usr/bin/env python3
"""Create character-disjoint phase-one splits and a font-held-out closed split."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import replace
import random
from pathlib import Path

from ocr_ids.dataset import SampleRecord, read_jsonl, write_jsonl
from ocr_ids.ids import parse_ids


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("output")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--zero-char-fraction", type=float, default=0.1)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--closed-per-character", type=int, default=1)
    return parser.parse_args()


def group_key(record: SampleRecord) -> str:
    return record.character or record.codepoint or record.sample_id.split("__", 1)[0]


def leaf_set(records: list[SampleRecord]) -> set[str]:
    return {leaf for record in records for leaf in parse_ids(record.ids).leaves()}


def select_zero_char_groups(
    groups: dict[str, list[SampleRecord]], target: int, rng: random.Random
) -> set[str]:
    """Select unseen whole characters while keeping every target leaf visible in training."""

    keys = list(groups)
    rng.shuffle(keys)
    leaf_frequency = Counter(
        leaf for key in keys for leaf in leaf_set(groups[key])
    )
    selected: set[str] = set()
    for key in keys:
        leaves = leaf_set(groups[key])
        if leaves and all(leaf_frequency[leaf] > 1 for leaf in leaves):
            selected.add(key)
            for leaf in leaves:
                leaf_frequency[leaf] -= 1
        if len(selected) >= target:
            break
    return selected


def select_validation_groups(
    groups: dict[str, list[SampleRecord]],
    candidates: list[str],
    protected_leaves: set[str],
    target: int,
    rng: random.Random,
) -> set[str]:
    """Hold out validation characters without hiding zero-shot test leaves from training."""

    keys = list(candidates)
    rng.shuffle(keys)
    remaining_frequency = Counter(
        leaf for key in keys for leaf in leaf_set(groups[key])
    )
    selected: set[str] = set()
    for key in keys:
        leaves = leaf_set(groups[key])
        # A protected leaf must retain at least one training character.
        if all(remaining_frequency[leaf] > 1 for leaf in leaves & protected_leaves):
            selected.add(key)
            for leaf in leaves:
                remaining_frequency[leaf] -= 1
        if len(selected) >= target:
            break
    return selected


def main() -> int:
    args = arguments()
    rng = random.Random(args.seed)
    records = list(read_jsonl(args.manifest))
    grouped: dict[str, list[SampleRecord]] = defaultdict(list)
    for record in records:
        grouped[group_key(record)].append(record)

    zero_target = round(len(grouped) * args.zero_char_fraction)
    zero_keys = select_zero_char_groups(grouped, zero_target, rng)
    remaining_keys = [key for key in grouped if key not in zero_keys]
    validation_count = round(len(grouped) * args.validation_fraction)
    protected_leaves = {
        leaf for key in zero_keys for leaf in leaf_set(grouped[key])
    }
    validation_keys = select_validation_groups(
        grouped,
        remaining_keys,
        protected_leaves,
        validation_count,
        rng,
    )
    train_keys = set(remaining_keys) - validation_keys

    train: list[SampleRecord] = []
    closed: list[SampleRecord] = []
    for key in sorted(train_keys):
        values = list(grouped[key])
        rng.shuffle(values)
        holdout = min(args.closed_per_character, max(0, len(values) - 1))
        closed.extend(replace(record, metadata={**record.metadata, "split": "test_closed"}) for record in values[:holdout])
        train.extend(replace(record, metadata={**record.metadata, "split": "train"}) for record in values[holdout:])

    validation = [
        replace(record, metadata={**record.metadata, "split": "validation"})
        for key in sorted(validation_keys)
        for record in grouped[key]
    ]
    zero_char = [
        replace(record, metadata={**record.metadata, "split": "test_zero_char"})
        for key in sorted(zero_keys)
        for record in grouped[key]
    ]
    output = Path(args.output)
    write_jsonl(train, output / "train.jsonl")
    write_jsonl(validation, output / "validation.jsonl")
    write_jsonl(closed, output / "test_closed.jsonl")
    write_jsonl(zero_char, output / "test_zero_char.jsonl")

    train_characters = {group_key(record) for record in train}
    zero_characters = {group_key(record) for record in zero_char}
    assert train_characters.isdisjoint(zero_characters)
    assert leaf_set(zero_char).issubset(leaf_set(train))
    print(
        f"train={len(train)} validation={len(validation)} "
        f"test_closed={len(closed)} test_zero_char={len(zero_char)}"
    )
    if len(zero_keys) < zero_target:
        print(f"warning: zero-char target={zero_target}, feasible={len(zero_keys)}")
    if len(validation_keys) < validation_count:
        print(f"warning: validation target={validation_count}, feasible={len(validation_keys)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
