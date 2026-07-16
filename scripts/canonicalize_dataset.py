#!/usr/bin/env python3
"""Build a conservative, versioned training corpus without manual annotation.

The first rule set intentionally does not infer equivalence between components.
Records outside strict Unicode IDS terminals are retained in a review manifest,
not silently rewritten or discarded.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path

from ocr_ids.canonical import is_han_target
from ocr_ids.dataset import SampleRecord, read_jsonl, write_jsonl
from ocr_ids.ids import canonicalize, parse_ids, validate_ids


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--rules", required=True)
    parser.add_argument("--output", required=True, help="Accepted canonical JSONL")
    parser.add_argument("--review-output", required=True, help="Excluded JSONL with reasons")
    parser.add_argument("--report", required=True)
    return parser.parse_args()


def main() -> int:
    args = arguments()
    rules = json.loads(Path(args.rules).read_text(encoding="utf-8"))
    version = rules["version"]
    leaf_aliases = rules.get("leaf_aliases", {})
    accepted: list[SampleRecord] = []
    review: list[SampleRecord] = []
    excluded = Counter()
    alternatives_kept = 0

    for record in read_jsonl(args.manifest):
        reasons: list[str] = []
        if not is_han_target(record.character):
            reasons.append("non_han_target")
        root = canonicalize(parse_ids(record.ids), leaf_aliases)
        ids = root.to_prefix()
        strict_problems = validate_ids(ids, strict_terminals=True)
        if strict_problems:
            reasons.append("non_strict_terminal")

        canonical_alternatives: list[str] = []
        for value in record.alternatives:
            alternative = canonicalize(parse_ids(value), leaf_aliases).to_prefix()
            if not validate_ids(alternative, strict_terminals=True) and alternative != ids:
                canonical_alternatives.append(alternative)
        canonical_alternatives = list(dict.fromkeys(canonical_alternatives))
        alternatives_kept += len(canonical_alternatives)
        metadata = {
            **record.metadata,
            "ids_raw": record.ids,
            "canonicalization_version": version,
            "canonicalization_rules": Path(args.rules).name,
        }
        transformed = replace(
            record,
            ids=ids,
            alternatives=canonical_alternatives,
            metadata=metadata,
        )
        if reasons:
            excluded.update(reasons)
            review.append(
                replace(transformed, metadata={**metadata, "review_reasons": reasons})
            )
        else:
            accepted.append(transformed)

    write_jsonl(accepted, args.output)
    write_jsonl(review, args.review_output)
    report = {
        "version": version,
        "input_manifest": str(Path(args.manifest).resolve()),
        "rules": str(Path(args.rules).resolve()),
        "accepted_records": len(accepted),
        "review_records": len(review),
        "excluded_by_reason": dict(sorted(excluded.items())),
        "accepted_alternative_ids": alternatives_kept,
        "leaf_alias_count": len(leaf_aliases),
        "policy": rules["description"],
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
