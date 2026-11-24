import argparse
import os
import sys
from pathlib import Path

# Prefer the active environment's eSpeak when Python is invoked without activation.
os.environ["PATH"] = str(Path(sys.executable).absolute().parent) + os.pathsep + os.environ.get("PATH", "")

from vits_common import (
    build_audio_config,
    build_dataset_config,
    build_vits_config,
    parse_resume_args,
    run_training,
)

DEFAULT_TEST_SENTENCES = [
    "It took me quite a long time to develop a voice, and now that I have it, I'm not going to be silent.",
    "Although the weather changed suddenly, she continued speaking calmly and clearly.",
    "Can you hear the difference between a question, a statement, and an exclamation?",
    "Dr. Smith bought three books for twelve dollars and fifty cents on January fifth.",
    "The quick brown fox jumps over the lazy dog.",
]


def parse_run_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the English VITS baseline.")
    parser.add_argument("--lang_tag", default="en", help="Language folder under the output root.")
    parser.add_argument("--dataset", default="en_ljspeech", help="Dataset folder under the data root.")
    parser.add_argument("--meta_file", default="metadata.csv", help="Metadata file inside the dataset folder.")
    parser.add_argument("--run_name", default="en_ljspeech_vits_real", help="Study name, also the output folder name.")
    parser.add_argument("--data_root", type=Path, default=None, help="Defaults to <AI_ROOT>/data.")
    parser.add_argument("--output_root", type=Path, default=None, help="Defaults to <AI_ROOT>/vits_runs.")
    parser.add_argument("--base_config", type=Path, default=None, help="Defaults to vits_engine/configs/vits_en.json.")
    parser.add_argument("--text_cleaner", default="english_cleaners")
    parser.add_argument("--phonemizer", default="espeak")
    parser.add_argument("--phoneme_language", default="en-us")
    parser.add_argument("--sample_rate", type=int, default=22050)
    parser.add_argument("--test_sentences", type=Path, default=None, help="Text file with one sentence per line.")
    args = parse_resume_args(parser)
    if args.sample_rate <= 0:
        parser.error("--sample_rate must be greater than 0")
    return args


def load_test_sentences(path: Path | None) -> list[list[str]]:
    if path is None:
        return [[line] for line in DEFAULT_TEST_SENTENCES]
    lines = [line.strip() for line in path.read_text(encoding="utf8").splitlines()]
    sentences = [[line] for line in lines if line]
    if not sentences:
        raise ValueError(f"No sentences in {path}")
    return sentences


args = parse_run_args()

AI_ROOT = Path(os.environ.get("AI_ROOT", str(Path(__file__).resolve().parents[3]))).resolve()
DATA_ROOT = (args.data_root or AI_ROOT / "data").resolve()
OUTPUT_ROOT = (args.output_root or AI_ROOT / "vits_runs").resolve()
BASE_CONFIG = (args.base_config or AI_ROOT / "vits_engine" / "configs" / "vits_en.json").resolve()

DATASET_PATH = str(DATA_ROOT / args.dataset)
OUTPUT_PATH = str(OUTPUT_ROOT / args.lang_tag / args.run_name)

dataset_config = build_dataset_config(
    dataset_name=args.dataset,
    dataset_path=DATASET_PATH,
    meta_file=args.meta_file,
    language=args.lang_tag,
)

audio_config = build_audio_config(sample_rate=args.sample_rate)

config = build_vits_config(
    run_name=args.run_name,
    output_path=OUTPUT_PATH,
    dataset_config=dataset_config,
    audio_config=audio_config,
    text_cleaner=args.text_cleaner,
    phonemizer=args.phonemizer,
    phoneme_language=args.phoneme_language,
    test_sentences=load_test_sentences(args.test_sentences),
    characters=None,
    continue_path=args.continue_path,
    restore_path=args.restore_path,
    restore_config=args.restore_config,
    base_config=str(BASE_CONFIG),
    epochs=args.epochs,
    batch_size=args.batch_size,
    eval_batch_size=args.eval_batch_size,
    num_workers=args.num_workers,
    num_eval_workers=args.num_eval_workers,
    lr_gen=args.lr_gen,
    lr_disc=args.lr_disc,
)

run_training(
    config=config,
    dataset_config=dataset_config,
    output_path=OUTPUT_PATH,
    continue_path=args.continue_path,
    restore_path=args.restore_path,
)
