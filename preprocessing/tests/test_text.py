import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ["PATH"] = str(Path(sys.executable).absolute().parent) + os.pathsep + os.environ.get("PATH", "")
from TTS.tts.utils.text.cleaners import english_cleaners
from TTS.config import load_config
from TTS.tts.utils.text.tokenizer import TTSTokenizer

class TextTests(unittest.TestCase):
    def test_cleaner(self):
        self.assertEqual(english_cleaners(""), "")
        self.assertEqual(english_cleaners("Dr. Smith has 12 books!"), "doctor smith has twelve books!")
        self.assertEqual(english_cleaners("Hello, WORLD."), "hello, world.")

    def test_checkpoint_config_tokenizes_with_espeak(self):
        config = load_config(ROOT / "vits_engine/configs/vits_en.json")
        tokenizer, _ = TTSTokenizer.init_from_config(config)
        self.assertTrue(tokenizer.use_phonemes)
        self.assertTrue(tokenizer.text_to_ids("Hello, world."))
        self.assertFalse(tokenizer.not_found_characters)
