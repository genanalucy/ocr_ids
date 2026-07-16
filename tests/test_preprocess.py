import unittest

from PIL import Image, ImageDraw

from ocr_ids.preprocess import normalize_single_glyph


class GlyphPreprocessTests(unittest.TestCase):
    def test_colored_foreground_becomes_square_black_on_white(self) -> None:
        image = Image.new("RGB", (40, 60), "white")
        ImageDraw.Draw(image).rectangle((10, 15, 27, 48), fill=(30, 100, 220))
        result = normalize_single_glyph(image)
        self.assertEqual(result.image.size, (224, 224))
        self.assertEqual(result.foreground_bbox, (10, 15, 28, 49))
        self.assertLess(result.image.getpixel((112, 112)), 150)
        self.assertGreater(result.image.getpixel((0, 0)), 240)

    def test_edge_touching_input_is_reported(self) -> None:
        image = Image.new("L", (20, 20), 255)
        ImageDraw.Draw(image).rectangle((0, 4, 8, 16), fill=0)
        result = normalize_single_glyph(image)
        self.assertIn("content_touches_input_edge", result.warnings)
