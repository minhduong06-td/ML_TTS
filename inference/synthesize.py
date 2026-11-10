"""Synthesize one or more English utterances with the local VITS checkpoint."""
import argparse
import os
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"vits_engine"))
os.environ["PATH"]=str(Path(sys.executable).absolute().parent)+os.pathsep+os.environ.get("PATH","")

def main():
    p=argparse.ArgumentParser(description=__doc__); group=p.add_mutually_exclusive_group(required=True); group.add_argument("--text"); group.add_argument("--sentences",type=Path)
    p.add_argument("--checkpoint",type=Path,required=True); p.add_argument("--config",type=Path,required=True); p.add_argument("--output",type=Path,required=True,help="WAV for --text or output directory for --sentences"); p.add_argument("--device",choices=("cpu","cuda"),default="cpu"); a=p.parse_args()
    if not a.checkpoint.is_file() or not a.config.is_file(): p.error("checkpoint and config must exist")
    from TTS.utils.synthesizer import Synthesizer
    import torch
    if a.device=="cuda" and not torch.cuda.is_available(): p.error("CUDA unavailable")
    texts=[a.text] if a.text is not None else [x.strip() for x in a.sentences.read_text(encoding="utf8").splitlines() if x.strip()]
    if not texts or any(not text.strip() for text in texts): p.error("no text supplied")
    outputs=[a.output] if a.text is not None else [a.output / f"{i:04d}.wav" for i in range(1,len(texts)+1)]
    if any(f.exists() for f in outputs): p.error("output already exists; refusing overwrite")
    synth=Synthesizer(tts_checkpoint=a.checkpoint,tts_config_path=a.config,use_cuda=a.device=="cuda")
    failures=0
    for text,output in zip(texts,outputs):
        try:
            wav=synth.tts(text,split_sentences=False); output.parent.mkdir(parents=True,exist_ok=True); synth.save_wav(wav,str(output)); print(f"OK {output}")
        except Exception as exc:
            failures+=1; print(f"FAIL {output}: {exc}",file=sys.stderr)
    return int(bool(failures))
if __name__=="__main__": raise SystemExit(main())
