#!/usr/bin/env python3
"""Convert the tab-separated cjkvi-ids files into project JSONL records."""

from __future__ import annotations

import argparse
from pathlib import Path
import re

from ocr_ids.dataset import SampleRecord, write_jsonl
from ocr_ids.ids import validate_ids


_REGION_SUFFIX = re.compile(r"^(.*?)(?:\[([A-Z]+)\])?$")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--source-version")
    return parser.parse_args()


def main() -> int:
    args = arguments()
    records: list[SampleRecord] = []
    skipped = 0
    with Path(args.input).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip() or line.startswith(";;;") or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3:
                skipped += 1
                continue
            codepoint, character, *raw_values = fields
            choices: list[tuple[str, str | None]] = []
            for raw in raw_values:
                match = _REGION_SUFFIX.match(raw.strip())
                if not match:
                    continue
                candidate, region = match.groups()
                if candidate and not validate_ids(candidate):
                    choices.append((candidate, region))
            if not choices:
                skipped += 1
                continue
            first, region = choices[0]
            records.append(
                SampleRecord(
                    sample_id=f"cjkvi-{codepoint.removeprefix('U+').lower()}-{line_number}",
                    character=character,
                    codepoint=codepoint,
                    ids=first,
                    glyph_region=region,
                    source="cjkvi-ids",
                    source_version=args.source_version,
                    alternatives=[value for value, _ in choices[1:] if value != first],
                    metadata={"source_line": line_number},
                )
            )
    write_jsonl(records, args.output)
    print(f"imported={len(records)} skipped={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

