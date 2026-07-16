import unittest

from ocr_ids.structure_scope import flat_structure_kind


class FlatStructureScopeTests(unittest.TestCase):
    def test_accepts_four_flat_structures(self) -> None:
        self.assertEqual(flat_structure_kind("⿰日月"), "left_right")
        self.assertEqual(flat_structure_kind("⿱艹田"), "top_bottom")
        self.assertEqual(flat_structure_kind("⿲日月木"), "left_middle_right")
        self.assertEqual(flat_structure_kind("⿳日月木"), "top_middle_bottom")

    def test_rejects_leaf_enclosure_and_nested_structures(self) -> None:
        self.assertIsNone(flat_structure_kind("日"))
        self.assertIsNone(flat_structure_kind("⿴囗玉"))
        self.assertIsNone(flat_structure_kind("⿰日⿱艹田"))
