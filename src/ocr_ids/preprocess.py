"""Robust single-glyph normalization for browser screenshots and scans."""

from __future__ import annotations

from dataclasses import dataclass
import statistics

from PIL import Image, ImageChops, ImageOps


@dataclass(frozen=True, slots=True)
class PreprocessResult:
    image: Image.Image
    source_size: tuple[int, int]
    foreground_bbox: tuple[int, int, int, int] | None
    foreground_fraction: float
    contrast: int
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "source_size": self.source_size,
            "foreground_bbox": self.foreground_bbox,
            "foreground_fraction": round(self.foreground_fraction, 4),
            "contrast": self.contrast,
            "warnings": self.warnings,
        }


def _border_background(gray: Image.Image) -> int:
    width, height = gray.size
    if width < 3 or height < 3:
        return int(statistics.median(gray.getdata()))
    pixels = gray.load()
    values = [pixels[x, 0] for x in range(width)]
    values += [pixels[x, height - 1] for x in range(width)]
    values += [pixels[0, y] for y in range(1, height - 1)]
    values += [pixels[width - 1, y] for y in range(1, height - 1)]
    return int(statistics.median(values))


def normalize_single_glyph(image: Image.Image, *, size: int = 224) -> PreprocessResult:
    """Convert a colored, variably-sized glyph crop to model-ready black on white.

    The border median is treated as background, so dark text on light backgrounds
    and light text on dark backgrounds follow the same path. The function does
    not invent missing strokes: edge-touching content is surfaced as a warning.
    """

    gray = image.convert("L")
    width, height = gray.size
    if not width or not height:
        raise ValueError("图片尺寸不能为零")
    background = _border_background(gray)
    difference = ImageChops.difference(gray, Image.new("L", gray.size, background))
    maximum = difference.getextrema()[1]
    threshold = max(8, int(maximum * 0.15))
    mask = difference.point(lambda value: 255 if value >= threshold else 0)
    bbox = mask.getbbox()
    warnings: list[str] = []
    if maximum < 20:
        warnings.append("low_contrast")
    if bbox is None:
        warnings.append("no_clear_foreground")
        canvas = Image.new("L", (size, size), color=255)
        return PreprocessResult(canvas, (width, height), None, 0.0, maximum, tuple(warnings))

    left, top, right, bottom = bbox
    box_width, box_height = right - left, bottom - top
    fraction = (box_width * box_height) / (width * height)
    if left == 0 or top == 0 or right == width or bottom == height:
        warnings.append("content_touches_input_edge")
    if box_width / max(box_height, 1) > 1.45:
        warnings.append("possibly_multiple_glyphs")
    if fraction < 0.04:
        warnings.append("very_small_foreground")

    margin = max(2, round(max(box_width, box_height) * 0.18))
    cropped = difference.crop(bbox)
    side = max(box_width, box_height) + margin * 2
    canvas = Image.new("L", (side, side), color=255)
    normalized_ink = ImageOps.invert(ImageOps.autocontrast(cropped))
    offset = ((side - box_width) // 2, (side - box_height) // 2)
    canvas.paste(normalized_ink, offset)
    normalized = canvas.resize((size, size), Image.Resampling.LANCZOS)
    return PreprocessResult(
        normalized,
        (width, height),
        bbox,
        fraction,
        maximum,
        tuple(warnings),
    )
