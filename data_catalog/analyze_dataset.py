"""Validate LJSpeech metadata and inspect WAV headers without decoding samples."""
import argparse
import json
import re
import wave
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def inspect_metadata(metadata, wav_dir, limit=None):
    metadata, wav_dir = Path(metadata), Path(wav_dir)
    counts = Counter(); issues = []; seen = set(); referenced = set(); valid = []
    try:
        with metadata.open(encoding="utf-8", errors="strict") as source:
            for number, line in enumerate(source, 1):
                if limit is not None and number > limit: break
                counts["records"] += 1
                cols = line.rstrip("\r\n").split("|"); errors = []
                if len(cols) != 3:
                    counts["invalid_columns"] += 1; errors.append("expected three columns")
                ident = cols[0].strip()
                if not re.fullmatch(r"[A-Za-z0-9_-]+", ident):
                    counts["invalid_ids"] += 1; errors.append("invalid ID")
                else:
                    if ident in seen: counts["duplicate_ids"] += 1; errors.append("duplicate ID")
                    seen.add(ident); referenced.add(ident + ".wav")
                    if not (wav_dir / (ident + ".wav")).is_file():
                        counts["missing_wavs"] += 1; errors.append("missing WAV")
                if len(cols) < 3 or not cols[1].strip() or not cols[2].strip():
                    counts["empty_text"] += 1; errors.append("missing text")
                if errors: issues.append({"line": number, "id": ident, "errors": errors})
                else: valid.append({"id": ident, "original_text": cols[1], "normalized_text": cols[2], "wav": "wavs/" + ident + ".wav"})
    except (OSError, UnicodeError) as exc:
        issues.append({"error": str(exc)}); counts["read_errors"] += 1
    counts["valid_records"] = len(valid)
    orphans = sorted(f.name for f in wav_dir.glob("*.wav") if f.name not in referenced) if limit is None else []
    counts["orphan_wavs"] = len(orphans)
    for name in ("records", "invalid_columns", "invalid_ids", "duplicate_ids", "missing_wavs", "empty_text", "read_errors", "orphan_wavs"):
        counts.setdefault(name, 0)
    return {"metadata":str(metadata), "counts":dict(counts), "issues":issues, "orphan_wavs":orphans, "valid":valid, "limited":limit is not None}


def inspect_wav(path):
    try:
        with wave.open(str(path),"rb") as f:
            frames=f.getnframes(); rate=f.getframerate()
            result={"path":str(path),"sample_rate":rate,"channels":f.getnchannels(),"frames":frames,"duration_seconds":frames/rate if rate else 0,"sample_width":f.getsampwidth()}
            if not frames or not rate: result["error"]="empty audio or zero sample rate"
            return result
    except (OSError, EOFError, wave.Error) as exc: return {"path":str(path),"error":str(exc)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=ROOT / "data/en_ljspeech/metadata.csv")
    parser.add_argument("--wav-dir", type=Path, default=ROOT / "data/en_ljspeech/wavs")
    parser.add_argument("--audio-sample", type=int, default=0, help="number of WAV headers; 0 checks all")
    parser.add_argument("--output", type=Path, help="optional JSON report path")
    args = parser.parse_args()
    if args.audio_sample < 0: parser.error("--audio-sample must be nonnegative")
    result = inspect_metadata(args.metadata, args.wav_dir)
    files = sorted(args.wav_dir.glob("*.wav"))
    checked = files[:args.audio_sample] if args.audio_sample else files
    headers = [inspect_wav(f) for f in checked]
    valid_headers = [h for h in headers if "error" not in h]
    durations = [h["duration_seconds"] for h in valid_headers]
    lengths = [len(row["normalized_text"]) for row in result["valid"]]
    bins = Counter("<1s" if d < 1 else "1-<5s" if d < 5 else "5-<10s" if d < 10 else ">=10s" for d in durations)
    mismatches = [h for h in valid_headers if h["sample_rate"] != 22050 or h["channels"] != 1]
    resolved_metadata = args.metadata.resolve()
    metadata_name = str(resolved_metadata.relative_to(ROOT)) if resolved_metadata.is_relative_to(ROOT) else str(resolved_metadata)
    report = {"metadata": metadata_name, "counts": result["counts"], "wav_files": len(files),
        "wav_headers_checked": len(headers), "header_errors": [h for h in headers if "error" in h], "audio_format_mismatches": mismatches,
        "sample_rates": dict(Counter(str(h["sample_rate"]) for h in valid_headers)),
        "channels": dict(Counter(str(h["channels"]) for h in valid_headers)),
        "duration_seconds": {"total":sum(durations), "min":min(durations,default=None), "max":max(durations,default=None), "bins":dict(bins)},
        "normalized_text_characters": {"min":min(lengths,default=None), "max":max(lengths,default=None), "mean":sum(lengths)/len(lengths) if lengths else None},
        "issues":result["issues"], "orphan_wavs":result["orphan_wavs"]}
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        target = args.output.resolve()
        if target.is_relative_to((ROOT / "data").resolve()) or target.is_relative_to((ROOT / "vits_runs").resolve()) or target == args.metadata.resolve() or target.is_relative_to(args.wav_dir.resolve()):
            parser.error("refusing to write inside dataset, artifact, or source input paths")
        target.parent.mkdir(parents=True, exist_ok=True); target.write_text(text + "\n", encoding="utf8")
    print(text)
    return int(bool(result["issues"] or result["orphan_wavs"] or report["header_errors"] or mismatches or not files))

if __name__ == "__main__":
    raise SystemExit(main())
