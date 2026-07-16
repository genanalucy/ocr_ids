#!/usr/bin/env python3
"""Render codepoint-labelled IDS records with one or more font files."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from fontTools.ttLib import TTCollection, TTFont
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from ocr_ids.dataset import SampleRecord, read_jsonl, write_jsonl


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True, help="输入 JSONL")
    parser.add_argument("--font", action="append", required=True, help="字体文件，可重复")
    parser.add_argument("--output", required=True)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--font-size", type=int, default=210)
    return parser.parse_args()


def parse_font_spec(value: str) -> tuple[Path, int]:
    path_value, separator, index_value = value.rpartition("#")
    if separator and index_value.isdigit():
        return Path(path_value), int(index_value)
    return Path(value), 0


def font_codepoints(path: Path, index: int = 0) -> set[int]:
    collection = None
    if path.suffix.lower() in {".ttc", ".otc"}:
        collection = TTCollection(path, lazy=True)
        if index >= len(collection.fonts):
            collection.close()
            raise ValueError(f"{path} 只有 {len(collection.fonts)} 个字体 face")
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


def render(character: str, font: ImageFont.FreeTypeFont, size: int) -> Image.Image:
    image = Image.new("L", (size, size), color=255)
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = draw.textbbox((0, 0), character, font=font)
    width, height = right - left, bottom - top
    position = ((size - width) / 2 - left, (size - height) / 2 - top)
    draw.text(position, character, font=font, fill=0)
    return image


def main() -> int:
    args = arguments()
    output = Path(args.output)
    image_dir = output / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    source_records = list(read_jsonl(args.labels))
    rendered: list[SampleRecord] = []

    for font_value in args.font:
        font_path, font_index = parse_font_spec(font_value)
        cmap = font_codepoints(font_path, font_index)
        pil_font = ImageFont.truetype(str(font_path), args.font_size, index=font_index)
        font_key = f"{font_path.stem.replace(' ', '_')}__face{font_index}"
        for source in tqdm(source_records, desc=font_key):
            if not source.character or ord(source.character) not in cmap:
                continue
            codepoint = f"u{ord(source.character):05x}"
            sample_id = f"{source.sample_id}__{font_key}"
            destination = image_dir / f"{codepoint}__{sample_id}.png"
            render(source.character, pil_font, args.size).save(destination)
            metadata = {
                **source.metadata,
                "font": str(font_path),
                "font_index": font_index,
                "render_size": args.size,
            }
            rendered.append(
                replace(
                    source,
                    sample_id=sample_id,
                    image_path=str(destination),
                    metadata=metadata,
                )
            )

    write_jsonl(rendered, output / "manifest.jsonl")
    print(f"rendered={len(rendered)} manifest={output / 'manifest.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
