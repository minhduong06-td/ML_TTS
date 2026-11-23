"""Extract real TensorBoard scalars and inspect synthesized audio."""
import argparse
import csv
import json
from pathlib import Path

def events(logdir,out):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out.mkdir(parents=True,exist_ok=True); summary=[]
    for index,path in enumerate(sorted(logdir.rglob("events.out.tfevents.*"))):
        accumulator=EventAccumulator(str(path),size_guidance={"scalars":0}); accumulator.Reload(); tags=accumulator.Tags().get("scalars",[])
        summary.append({"source":str(path),"tags":tags}); folder=out/f"event_{index:02d}"; folder.mkdir(exist_ok=True)
        with (folder/"scalars.csv").open("w",newline="",encoding="utf8") as f:
            w=csv.writer(f);w.writerow(["source","tag","step","wall_time","value"])
            for tag in tags:
                values=accumulator.Scalars(tag)
                for v in values: w.writerow([str(path),tag,v.step,v.wall_time,v.value])
                if any(x in tag.lower() for x in ("loss","learning_rate","lr_")) and values:
                    fig,ax=plt.subplots(); ax.plot([v.step for v in values],[v.value for v in values]);ax.set(xlabel="Step",ylabel=tag,title=tag);fig.tight_layout(); fig.savefig(folder/(f"tag_{tags.index(tag):03d}.png"));plt.close(fig)
    (out/"sources.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf8");print(json.dumps(summary,indent=2));return int(not summary)

def audio(path):
    import numpy as np
    import soundfile as sf
    samples,rate=sf.read(path,always_2d=True); peak=float(np.max(np.abs(samples))) if samples.size else 0; rms=float(np.sqrt(np.mean(samples**2))) if samples.size else 0
    result={"path":str(path),"sample_rate":rate,"channels":samples.shape[1],"duration_seconds":len(samples)/rate,"peak":peak,"rms":rms,"samples_at_or_above_0_999":int(np.count_nonzero(np.abs(samples)>=0.999)),"near_silence":rms<1e-4};print(json.dumps(result,indent=2)); return int(not samples.size or not np.isfinite(samples).all())

def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest="cmd",required=True); e=sub.add_parser("events");e.add_argument("logdir",type=Path);e.add_argument("output",type=Path); a=sub.add_parser("audio");a.add_argument("wav",type=Path); args=p.parse_args();return events(args.logdir,args.output) if args.cmd=="events" else audio(args.wav)
if __name__=="__main__": raise SystemExit(main())
