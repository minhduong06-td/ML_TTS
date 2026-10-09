import tempfile
import unittest
import wave
from pathlib import Path
from data_catalog.analyze_dataset import inspect_wav

class AudioTests(unittest.TestCase):
    def test_header_and_invalid_file(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"a.wav"
            with wave.open(str(p),"wb") as f:
                f.setparams((1,2,22050,0,"NONE","not compressed")); f.writeframes(b"\0\0"*100)
            self.assertEqual(inspect_wav(p)["sample_rate"],22050)
            p.write_bytes(b"broken")
            self.assertIn("error",inspect_wav(p))
