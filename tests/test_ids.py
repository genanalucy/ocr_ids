import unittest

from ocr_ids.ids import IDSParseError, canonicalize, parse_ids, tokenize_ids, validate_ids


class IDSParserTest(unittest.TestCase):
    def test_binary_ids(self):
        node = parse_ids("⿰日月")
        self.assertEqual(node.to_prefix(), "⿰日月")
        self.assertEqual(node.leaves(), ("日", "月"))
        self.assertEqual(node.depth, 2)
        self.assertEqual(node.to_dict(), {"operator": "⿰", "children": ["日", "月"]})

    def test_nested_and_ternary_ids(self):
        node = parse_ids("⿱艹⿲口日月")
        self.assertEqual(node.leaves(), ("艹", "口", "日", "月"))
        self.assertEqual(node.depth, 3)

    def test_incomplete_and_trailing_ids_fail(self):
        with self.assertRaises(IDSParseError):
            parse_ids("⿰日")
        with self.assertRaises(IDSParseError):
            parse_ids("⿰日月木")

    def test_cdp_entity_is_one_token(self):
        self.assertEqual(tokenize_ids("⿰木&CDP-8BF5;"), ("⿰", "木", "&CDP-8BF5;"))

    def test_strict_terminal_validation(self):
        self.assertEqual(validate_ids("⿰日？", strict_terminals=True), ())
        self.assertTrue(validate_ids("⿰日A", strict_terminals=True))

    def test_explicit_canonicalization(self):
        result = canonicalize(parse_ids("⿰亻木"), {"亻": "人"})
        self.assertEqual(result.to_prefix(), "⿰人木")


if __name__ == "__main__":
    unittest.main()

