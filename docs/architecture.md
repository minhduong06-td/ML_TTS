# Existing architecture

The implementation is in `vits_engine/TTS/tts/models/vits.py` and related VITS layers. It includes a text encoder, posterior encoder, normalizing flow, monotonic alignment, stochastic duration prediction, waveform decoder and periodic/scale discriminators. Training combines duration, KL, mel reconstruction, generator, discriminator and feature matching objectives. Inference derives durations from text and generates audio without the training posterior or adversarial update. See the source for exact implementation; no model tensors or architecture were changed here.
