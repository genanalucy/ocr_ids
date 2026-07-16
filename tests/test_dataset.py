import tempfile
from pathlib import Path
import unittest

from ocr_ids.dataset import SampleRecord, read_jsonl, write_jsonl
from ocr_ids.vocab import build_vocabulary


class DatasetTest(unittest.TestCase):
    def test_jsonl_roundtrip(self):
        record = SampleRecord(
            sample_id="u660e-g",
            character="明",
            codepoint="U+660E",
            ids="⿰日月",
            alternatives=["⿰日月"],
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            write_jsonl([record], path)
            loaded = list(read_jsonl(path))
        self.assertEqual(loaded[0].sample_id, record.sample_id)
        self.assertEqual(loaded[0].ids, "⿰日月")

    def test_vocabulary_prefix_order(self):
        vocabulary = build_vocabulary(["⿰日月", "⿱艹田"])
        encoded = vocabulary.encode("⿰日月")
        mapping = vocabulary.token_to_id
        self.assertEqual(
            encoded,
            [mapping["<bos>"], mapping["⿰"], mapping["日"], mapping["月"], mapping["<eos>"]],
        )


if __name__ == "__main__":
    unittest.main()

