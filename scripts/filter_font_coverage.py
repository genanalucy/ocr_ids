#!/usr/bin/env python3
"""Select records renderable by a required number of local font faces."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path

from fontTools.ttLib import TTCollection, TTFont

from ocr_ids.dataset import read_jsonl, write_jsonl


def parse_font_spec(value: str) -> tuple[Path, int]:
    path_value, separator, index_value = value.rpartition("#")
    if separator and index_value.isdigit():
        return Path(path_value), int(index_value)
    return Path(value), 0


def cmap(path: Path, index: int) -> set[int]:
    collection = None
    if path.suffix.lower() in {".ttc", ".otc"}:
        collection = TTCollection(path, lazy=True)
        font = collection.fonts[index]
    else:
        font = TTFont(path, lazy=True)
    try:
        return {codepoint for table in font["cmap"].tables for codepoint in table.cmap}
    finally:
        if collection is not None:
            collection.close()
        else:
            font.close()


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--font", action="append", required=True)
    parser.add_argument("--min-fonts", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--review-output", required=True)
    parser.add_argument("--report", required=True)
    return parser.parse_args()


def main() -> int:
    args = arguments()
    font_specs = [parse_font_spec(value) for value in args.font]
    if not 1 <= args.min_fonts <= len(font_specs):
        raise ValueError("min-fonts 必须位于 1 和字体数量之间")
    coverage = [cmap(path, index) for path, index in font_specs]
    accepted = []
    review = []
    histogram = Counter()
    for record in read_jsonl(args.manifest):
        available = sum(
            bool(record.character and ord(record.character) in available_codepoints)
            for available_codepoints in coverage
        )
        histogram[available] += 1
        metadata = {
            **record.metadata,
            "font_coverage": available,
            "font_coverage_required": args.min_fonts,
        }
        if available >= args.min_fonts:
            accepted.append(replace(record, metadata=metadata))
        else:
            review.append(
                replace(
                    record,
                    metadata={
                        **metadata,
                        "review_reasons": [f"font_coverage:{available}/{len(font_specs)}"],
                    },
                )
            )
    write_jsonl(accepted, args.output)
    write_jsonl(review, args.review_output)
    report = {
        "input_records": len(accepted) + len(review),
        "renderable_records": len(accepted),
        "review_records": len(review),
        "font_specs": args.font,
        "min_fonts": args.min_fonts,
        "coverage_histogram": dict(sorted(histogram.items())),
    }
    destination = Path(args.report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

