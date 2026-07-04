#!/usr/bin/env python3
"""
denoise_dataset.py
------------------
Remove background noise from a folder of audio clips (one cleaned file out per
input file). Used before slicing/training so the cloned voice never learns the
room noise/hiss.

Denoiser priority (robust, never hard-fails):
  1. FRCRN  - neural speech enhancement (modelscope), best quality if available
  2. noisereduce - pure-Python spectral gate, always works (a listed dependency)

Usage:
  python src/denoise_dataset.py --input data/raw --output data/raw_clean
  python src/denoise_dataset.py --input data/raw --output data/raw_clean --method noisereduce
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from console import banner, step, info, ok, warn, err, value, progress_bar  # noqa: E402

AUDIO_EXTS = {".ogg", ".wav", ".mp3", ".m4a", ".flac", ".aac", ".opus", ".wma"}


def _list_audio(d: str):
    p = Path(d)
    if not p.is_dir():
        err(f"input folder not found: {d}")
        raise SystemExit(2)
    return sorted(f for f in p.iterdir() if f.suffix.lower() in AUDIO_EXTS)


def _load_frcrn():
    """Return an FRCRN pipeline, or None if modelscope isn't fully installed."""
    try:
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks

        model = "damo/speech_frcrn_ans_cirm_16k"
        return pipeline(Tasks.acoustic_noise_suppression, model=model)
    except Exception as e:  # noqa: BLE001
        warn(f"FRCRN unavailable ({type(e).__name__}: {str(e)[:100]}) -> falling back to noisereduce")
        return None


def _denoise_noisereduce(src: Path, dst: Path) -> None:
    import numpy as np
    import soundfile as sf
    import noisereduce as nr
    from pydub import AudioSegment

    audio = AudioSegment.from_file(src).set_channels(1)
    sr = audio.frame_rate
    y = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0
    reduced = nr.reduce_noise(y=y, sr=sr, stationary=True, prop_decrease=0.9)
    sf.write(str(dst), reduced, sr)


def main() -> int:
    ap = argparse.ArgumentParser(description="Denoise a folder of audio clips.")
    ap.add_argument("--input", required=True, help="Folder of source clips (e.g. data/raw).")
    ap.add_argument("--output", required=True, help="Folder for cleaned WAVs.")
    ap.add_argument("--method", choices=["auto", "frcrn", "noisereduce"], default="auto")
    args = ap.parse_args()

    banner("Denoise dataset")
    files = _list_audio(args.input)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    value("clips", len(files))
    value("requested method", args.method)

    ans = None
    if args.method in ("auto", "frcrn"):
        ans = _load_frcrn()
        if ans is None and args.method == "frcrn":
            err("FRCRN explicitly requested but not available. Install modelscope deps or use --method noisereduce.")
            return 2

    active = "frcrn" if ans is not None else "noisereduce"
    ok(f"denoising with: {active}")

    done = 0
    for i, f in enumerate(files, 1):
        dst = out / (f.stem + ".wav")
        try:
            if ans is not None:
                ans(str(f), output_path=str(dst))
            else:
                _denoise_noisereduce(f, dst)
            done += 1
        except Exception as e:  # noqa: BLE001
            err(f"failed {f.name}: {str(e)[:120]}")
        progress_bar(i, len(files), prefix="denoise")

    print()
    if done == 0:
        err("No files were denoised.")
        return 2
    ok(f"wrote {done}/{len(files)} cleaned clips to {out}")
    info(f"Next: python src/prepare_dataset.py --input-dir {out} --output data/dataset/speaker1_clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
