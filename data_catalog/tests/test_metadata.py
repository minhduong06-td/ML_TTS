import tempfile
import unittest
from pathlib import Path
from data_catalog.analyze_dataset import inspect_metadata

class MetadataTests(unittest.TestCase):
    def test_invalid_duplicate_missing_and_columns(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d); (p/"wavs").mkdir(); (p/"wavs/A.wav").touch()
            (p/"m.csv").write_text("A|original|normalized\nA|again|again\nB|missing|missing\nbroken\n", encoding="utf8")
            result=inspect_metadata(p/"m.csv",p/"wavs")
            self.assertEqual(result["counts"]["duplicate_ids"],1)
            self.assertEqual(result["counts"]["missing_wavs"],2)
            self.assertEqual(result["counts"]["invalid_columns"],1)
            self.assertEqual(result["counts"]["valid_records"],1)
