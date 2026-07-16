import unittest

from ocr_ids.structure_scope import FLAT_STRUCTURE_OPERATORS


class FlatInferenceScopeTests(unittest.TestCase):
    def test_only_four_root_operators_are_in_scope(self) -> None:
        self.assertEqual(set(FLAT_STRUCTURE_OPERATORS), {"⿰", "⿱", "⿲", "⿳"})
