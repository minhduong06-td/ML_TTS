# TTS English baseline

English VITS baseline trained on LJSpeech. Everything here runs on CPU.

## Files

```
data/en_ljspeech/metadata.csv        id|text|normalized, 13,100 lines
vits_engine/TTS/                     VITS source used by the checkpoint
vits_engine/configs/vits_en.json     baseline config
vits_engine/recipes/local/           training entry point
inference/synthesize.py              text to WAV
evaluation/evaluate.py               TensorBoard export, WAV checks
```

WAV files, the checkpoint and the phoneme cache stay local. The two `.txt`
files in `vits_runs/en/en_ljspeech_vits_real/` are the Drive links for them.

## Setup

Python 3.11 with torch 2.5.1 CPU. The venv sits in `tts/`:

```bash
source tts/bin/activate
```

`espeak-ng` has to be on PATH. The venv already ships it in `tts/bin`.
Package versions are pinned in `requirements.txt`.

## Check the dataset

```bash
./tts/bin/python -m data_catalog.analyze_dataset
```

Reads the metadata, checks the IDs and reads the WAV headers. Takes about 8
seconds for the full corpus.

## Train

```bash
./tts/bin/python vits_engine/recipes/local/train_vits_en.py
```

Writes a new run under `vits_runs/en/en_ljspeech_vits_real`. The checkpoint we
use comes from an April run that was made before this repository, so training
is not needed to reproduce our numbers.

No checkpoint path is stored in the config. To fine-tune or resume, pass it on
the command line with `--restore_path` (plus `--restore_config` if the config is
not next to the checkpoint) or `--continue_path`. Those paths were not tested.

## Synthesize

```bash
run=vits_runs/en/en_ljspeech_vits_real/en_ljspeech_vits_real-November-23-2026_11+57PM-0000000
./tts/bin/python inference/synthesize.py \
  --text "Hello, this is a test of the English VITS model." \
  --checkpoint $run/best_model.pth \
  --config $run/config.json \
  --output vits_runs/en/evaluation/hello.wav \
  --device cpu
```

For a batch, pass `--sentences file.txt` and give `--output` a directory. The
script refuses to overwrite an existing WAV.

## Evaluate

```bash
./tts/bin/python evaluation/evaluate.py audio vits_runs/en/evaluation/hello.wav
./tts/bin/python evaluation/evaluate.py events vits_runs/en vits_runs/en/evaluation/tensorboard
```

`events` writes one CSV per TensorBoard file plus plots for loss and learning
rate tags. Keep that output out of git.

## Tests

```bash
./tts/bin/python -m unittest preprocessing.tests.test_text \
  data_catalog.tests.test_metadata data_catalog.tests.test_audio
```

## License

`vits_engine/TTS` is Coqui TTS code, MPL-2.0. There was no license file in this
workspace, so check the upstream terms before publishing anything.
