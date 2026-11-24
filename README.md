# VITS text-to-speech

VITS is a single-stage end-to-end TTS model: one network maps phonemes to a
waveform, with no separate acoustic model and vocoder. Training combines a
conditional VAE over the mel spectrogram, a normalizing flow between the latent
spaces, monotonic alignment search and a GAN discriminator, so the text encoder,
posterior encoder, flow and HiFi-GAN decoder are learned together.

The baseline in this repository:

- input: text -> `english_cleaners` -> eSpeak phonemes (`en-us`) -> IPA ids (117 symbols)
- text encoder: 6 layers, 2 attention heads, 768 FFN channels
- alignment and duration: monotonic alignment search in training, stochastic duration predictor (`use_sdp`) at inference
- decoder: HiFi-GAN at 22,050 Hz, FFT/window 1024, hop 256, 80 mel bands, 32-frame segments for the discriminators
- losses: mel 45.0, KL 1.0, duration 1.0, adversarial and feature matching 1.0

At inference the model takes durations from the duration predictor, drops the
posterior encoder and the adversarial update, and decodes the waveform directly.
It is a single-speaker model (`num_speakers = 0`): no speaker embedding, one
voice per checkpoint.

## Requirements

- Linux or WSL with bash. The commands below assume the repository sits at
  `~/vits`, but nothing depends on that path.
- Python 3.11 for the model. The virtualenv in `tts/` is 3.11.18 with the pinned
  versions from `requirements.txt`: torch and torchaudio 2.5.1 (CPU build),
  coqui-tts-trainer 0.5.0, numpy, librosa, soundfile, tensorboard.
- `espeak-ng` has to be on `PATH`. The environment ships a wrapper as
  `tts/bin/espeak-ng`, and the scripts add that directory by themselves.
- The web demo is optional and runs on a second interpreter with `fastapi`,
  `uvicorn`, `pydantic` and `python-multipart` (`web/be/requirements.txt`).

## Demo samples

| Sentence | Recording |
|---|---|
| Hello everybody, we are team 11 | [`en_20261124_095356_76166314.wav`](https://github.com/minhduong06-td/ML_TTS/raw/main/assets/en_20261124_095356_76166314.wav) |
| Ha Noi university of science and technology is the best university of the world | [`en_20261124_095646_8e8c01fb.wav`](https://github.com/minhduong06-td/ML_TTS/raw/main/assets/en_20261124_095646_8e8c01fb.wav) |

To record new samples, start the local web demo and open it in the browser:

```bash
USE_CUDA=0 VITS_PYTHON="$PWD/tts/bin/python" ./web/run_web.sh
# then open http://127.0.0.1:8000
```

### VITS controls

Three parameters go from the page (or the CLI) to the model. The values used
for the samples above are the middle column:

| Control | Used | Range in the page | Effect |
|---|---|---|---|
| `length_scale` | `1.0` | 0.70 - 1.80, step 0.05 | Duration scaling. Above 1.0 the voice gets slower and the clip longer, below 1.0 faster. |
| `noise_scale` | `0.67` | 0.10 - 1.20, step 0.01 | Randomness of the acoustic sampling. Low values give a flatter, more monotone voice, high values more variation and more risk of artefacts. |
| `noise_scale_dp` | `0.8` | 0.10 - 1.20, step 0.05 | Randomness of the duration predictor. It changes the rhythm and the pauses, not the timbre. |

```bash
run=vits_runs/en/en_ljspeech_vits_real/en_ljspeech_vits_real-November-23-2026_11+57PM-0000000
./tts/bin/python -m TTS.bin.synthesize \
  --text "Hello everybody, we are team 11" \
  --model_path "$run/best_model.pth" --config_path "$run/config.json" \
  --out_path vits_runs/en/evaluation/team11_slow.wav \
  --length_scale 1.2 --noise_scale 0.6 --noise_scale_dp 0.8 --device cpu
```

## Commands

### Train from scratch

```bash
./tts/bin/python vits_engine/recipes/local/train_vits_en.py \
  --run_name en_ljspeech_vits_new
```

- Writes `vits_runs/en/en_ljspeech_vits_new`. 
- Defaults read: `data/en_ljspeech/metadata.csv`. 
- Change that with `--data_root`, `--dataset`, `--meta_file`. 
- Other flags: `--epochs`, `--batch_size`, `--lr_gen`, `--lr_disc`, `--sample_rate`, `--text_cleaner`, `--phonemizer`, `--phoneme_language`, `--test_sentences`.

### Restore, fine-tune from a checkpoint

```bash
run=vits_runs/en/en_ljspeech_vits_real/en_ljspeech_vits_real-November-23-2026_11+57PM-0000000
./tts/bin/python vits_engine/recipes/local/train_vits_en.py \
  --restore_path "$run/best_model.pth" \
  --restore_config "$run/config.json" \
  --run_name en_ljspeech_vits_finetune
```

### Continue, resume the same run

```bash
run=vits_runs/en/en_ljspeech_vits_real/en_ljspeech_vits_real-November-23-2026_11+57PM-0000000
./tts/bin/python vits_engine/recipes/local/train_vits_en.py --continue_path "$run"
```

### Generate a WAV

```bash
run=vits_runs/en/en_ljspeech_vits_real/en_ljspeech_vits_real-November-23-2026_11+57PM-0000000
./tts/bin/python inference/synthesize.py \
  --text "Hello everybody, we are team 11" \
  --checkpoint "$run/best_model.pth" \
  --config "$run/config.json" \
  --output vits_runs/en/evaluation/team11.wav \
  --device cpu
```