import unittest

from ocr_ids.canonical import is_han_target


class CanonicalizeTest(unittest.TestCase):
    def test_han_target_ranges(self):
        self.assertTrue(is_han_target("明"))
        self.assertTrue(is_han_target("𠀀"))
        self.assertFalse(is_han_target("α"))
        self.assertFalse(is_han_target("⺁"))


if __name__ == "__main__":
    unittest.main()
