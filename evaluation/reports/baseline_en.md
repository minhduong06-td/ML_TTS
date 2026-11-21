# English baseline: existing run

The checkpoint and events predate this work and were not produced by a new training run. The checkpoint's original config specifies 22,050 Hz, FFT/window 1024, hop 256, 80 mel bands, `english_cleaners`, eSpeak `en-us`, and its original IPA vocabulary. The baseline config preserves audio, model arguments, characters and phonemizer; only dataset/output/cache paths differ.

On Python 3.11.18 and CPU PyTorch 2.5.1, the local VITS model loaded the existing checkpoint with its default `strict=True`, and generated `vits_runs/en/evaluation/hello.wav` for "Hello, this is a test of the English VITS model." The WAV is mono, 22,050 Hz, 3.995 s, peak 0.999939, RMS 0.1372; one sample met the 0.999 full-scale threshold.
