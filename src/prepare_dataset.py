#!/usr/bin/env python3
"""
prepare_dataset.py
------------------
Turn raw voice clips into clean segments ready for GPT-SoVITS fine-tuning.

Accepts either:
  --input      one long recording (e.g. a 30-min file), OR
  --input-dir  a folder of clips (e.g. data/raw from download_dataset.py)

Pipeline per source:
  1. Decode (any ffmpeg format: ogg/wav/mp3/m4a/flac...).
  2. Convert to mono 32 kHz, normalize peak.
  3. Optional light denoise (--denoise).
  4. Slice on silence into ~3-10s utterances.
Segments are written sequentially to --output.

Usage:
  python src/prepare_dataset.py --input-dir data/raw --output data/dataset/speaker1
  python src/prepare_dataset.py --input my_30min.mp3 --output data/dataset/speaker1 --denoise

Requirements:
  pip install -r requirements.txt   (colorama, pydub, soundfile, numpy[, noisereduce])
  system ffmpeg:  apt-get install -y ffmpeg
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from console import banner, step, info, ok, warn, err, value, progress_bar  # noqa: E402

try:
    import numpy as np
    from pydub import AudioSegment, silence
except ImportError as e:
    err(f"Missing dependency: {getattr(e, 'name', e)}")
    err("Install with: pip install -r requirements.txt  (and apt-get install -y ffmpeg)")
    raise SystemExit(1)

TARGET_SR = 32000
AUDIO_EXTS = {".ogg", ".wav", ".mp3", ".m4a", ".flac", ".aac", ".opus", ".wma"}


def to_mono_32k(audio: AudioSegment) -> AudioSegment:
    audio = audio.set_channels(1).set_frame_rate(TARGET_SR)
    audio = audio.apply_gain(-1.0 - audio.max_dBFS)  # peak ~ -1 dBFS
    return audio


def maybe_denoise(audio: AudioSegment) -> AudioSegment:
    try:
        import noisereduce as nr
    except ImportError:
        warn("noisereduce not installed, skipping denoise (pip install noisereduce)")
        return audio
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    reduced = nr.reduce_noise(y=samples, sr=audio.frame_rate, stationary=True, prop_decrease=0.75)
    reduced = np.clip(reduced, -32768, 32767).astype(np.int16)
    return audio._spawn(reduced.tobytes())


def slice_on_silence(audio: AudioSegment, min_len: float, max_len: float):
    min_ms, max_ms = int(min_len * 1000), int(max_len * 1000)
    nonsilent = silence.detect_nonsilent(
        audio, min_silence_len=300, silence_thresh=audio.dBFS - 16, seek_step=10
    )
    if not nonsilent:
        return []

    merged = []
    buf_start, buf_end = nonsilent[0]
    for start, end in nonsilent[1:]:
        if (end - buf_start) <= max_ms:
            buf_end = end
        else:
            merged.append((buf_start, buf_end))
            buf_start, buf_end = start, end
    merged.append((buf_start, buf_end))

    padded = []
    for start, end in merged:
        s = max(0, start - 100)
        e = min(len(audio), end + 200)
        if (e - s) <= max_ms:
            padded.append((s, e))
        else:
            cur = s
            while cur < e:
                padded.append((cur, min(cur + max_ms, e)))
                cur += max_ms
    return [(s, e) for (s, e) in padded if (e - s) >= min_ms]


def collect_sources(args) -> list[Path]:
    if args.input:
        p = Path(args.input)
        if not p.is_file():
            err(f"--input not found: {p}")
            raise SystemExit(1)
        return [p]
    d = Path(args.input_dir)
    if not d.is_dir():
        err(f"--input-dir not found: {d}")
        raise SystemExit(1)
    files = sorted(f for f in d.iterdir() if f.suffix.lower() in AUDIO_EXTS)
    if not files:
        err(f"No audio files in {d} (looked for {sorted(AUDIO_EXTS)})")
        raise SystemExit(1)
    return files


def main() -> int:
    ap = argparse.ArgumentParser(description="Prepare voice clips for GPT-SoVITS fine-tuning.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--input", help="A single source recording.")
    g.add_argument("--input-dir", help="A folder of source clips (e.g. data/raw).")
    ap.add_argument("--output", required=True, help="Output dir for sliced WAV segments.")
    ap.add_argument("--min-len", type=float, default=3.0)
    ap.add_argument("--max-len", type=float, default=10.0)
    ap.add_argument("--denoise", action="store_true", help="Apply light noise reduction.")
    ap.add_argument("--prefix", default="seg")
    args = ap.parse_args()

    banner("Dataset preparation")
    sources = collect_sources(args)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    value("Sources", len(sources))
    value("Output dir", out)
    value("Segment length", f"{args.min_len}-{args.max_len}s")
    value("Denoise", args.denoise)
    print()

    seg_index = 0
    total_s = 0.0
    for i, src in enumerate(sources, 1):
        step(i, len(sources), src.name)
        try:
            audio = AudioSegment.from_file(src)
        except Exception as e:  # noqa: BLE001
            err(f"could not decode {src.name}: {e}")
            continue
        audio = to_mono_32k(audio)
        if args.denoise:
            audio = maybe_denoise(audio)

        ranges = slice_on_silence(audio, args.min_len, args.max_len)
        if not ranges:
            warn(f"no usable speech in {src.name} (too quiet/short?)")
            continue

        for j, (s, e) in enumerate(ranges, 1):
            seg_index += 1
            (out / f"{args.prefix}_{seg_index:04d}.wav").write_bytes(b"")  # touch for ordering
            audio[s:e].export(out / f"{args.prefix}_{seg_index:04d}.wav", format="wav")
            total_s += (e - s) / 1000.0
            progress_bar(j, len(ranges), prefix="slicing")

    print()
    if seg_index == 0:
        err("No segments produced. Lower --min-len or check the recordings.")
        return 2
    ok(f"Wrote {seg_index} segments to {out}")
    value("Total usable speech", f"{total_s/60:.1f} min")
    value("Avg segment", f"{total_s/seg_index:.1f}s")
    info("Next: open the GPT-SoVITS WebUI, run ASR on this folder, then train (docs/TRAINING.md).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
