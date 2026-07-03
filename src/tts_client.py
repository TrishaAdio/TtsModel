#!/usr/bin/env python3
"""
tts_client.py
-------------
Thin client for the GPT-SoVITS API server (api_v2.py). Import this into the AI
you built to turn text into speech in your cloned voice.

Start the server on the VPS first:
    python api_v2.py -a 127.0.0.1 -p 9880

Then from your AI:
    from tts_client import VoiceTTS
    tts = VoiceTTS(
        base_url="http://127.0.0.1:9880",
        ref_audio_path="/abs/path/data/dataset/speaker1/seg_0007.wav",
        prompt_text="exact words spoken in that reference clip",
        prompt_lang="en", text_lang="en",
    )
    audio = tts.synthesize("Hello, this is my cloned voice.")
    tts.say("Saved to a file too.", out_path="reply.wav")
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import requests


@dataclass
class VoiceTTS:
    base_url: str = "http://127.0.0.1:9880"
    ref_audio_path: str = ""
    prompt_text: str = ""
    prompt_lang: str = "en"
    text_lang: str = "en"
    top_k: int = 15
    top_p: float = 1.0
    temperature: float = 0.95
    speed_factor: float = 1.0
    text_split_method: str = "cut5"
    batch_size: int = 1
    timeout: int = 120
    session: requests.Session = field(default_factory=requests.Session, repr=False)

    def set_weights(self, gpt_weights_path: str, sovits_weights_path: str) -> None:
        """Point the running server at your fine-tuned checkpoints."""
        r1 = self.session.get(
            f"{self.base_url}/set_gpt_weights",
            params={"weights_path": gpt_weights_path}, timeout=self.timeout,
        )
        r1.raise_for_status()
        r2 = self.session.get(
            f"{self.base_url}/set_sovits_weights",
            params={"weights_path": sovits_weights_path}, timeout=self.timeout,
        )
        r2.raise_for_status()

    def synthesize(self, text: str, **overrides) -> bytes:
        """Return WAV bytes for `text`. Raises RuntimeError on server error."""
        if not self.ref_audio_path or not self.prompt_text:
            raise ValueError("ref_audio_path and prompt_text must be set before synthesis.")
        payload = {
            "text": text,
            "text_lang": overrides.get("text_lang", self.text_lang),
            "ref_audio_path": overrides.get("ref_audio_path", self.ref_audio_path),
            "prompt_text": overrides.get("prompt_text", self.prompt_text),
            "prompt_lang": overrides.get("prompt_lang", self.prompt_lang),
            "top_k": overrides.get("top_k", self.top_k),
            "top_p": overrides.get("top_p", self.top_p),
            "temperature": overrides.get("temperature", self.temperature),
            "speed_factor": overrides.get("speed_factor", self.speed_factor),
            "text_split_method": overrides.get("text_split_method", self.text_split_method),
            "batch_size": overrides.get("batch_size", self.batch_size),
            "media_type": "wav",
            "streaming_mode": False,
        }
        resp = self.session.post(f"{self.base_url}/tts", json=payload, timeout=self.timeout)
        if resp.status_code != 200:
            try:
                detail = resp.json()
            except json.JSONDecodeError:
                detail = resp.text
            raise RuntimeError(f"TTS failed ({resp.status_code}): {detail}")
        return resp.content

    def say(self, text: str, out_path: str = "output.wav", **overrides) -> str:
        with open(out_path, "wb") as f:
            f.write(self.synthesize(text, **overrides))
        return out_path

    def wait_until_ready(self, retries: int = 30, delay: float = 2.0) -> bool:
        for _ in range(retries):
            try:
                self.session.get(f"{self.base_url}/", timeout=5)
                return True
            except requests.RequestException:
                time.sleep(delay)
        return False


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Quick test of the GPT-SoVITS TTS client.")
    ap.add_argument("--url", default="http://127.0.0.1:9880")
    ap.add_argument("--ref", required=True, help="Reference audio path (5-10s clean clip).")
    ap.add_argument("--ref-text", required=True, help="Exact transcript of the reference clip.")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--text", default="Hi, this is a test of my cloned voice.")
    ap.add_argument("--out", default="output.wav")
    args = ap.parse_args()

    tts = VoiceTTS(
        base_url=args.url, ref_audio_path=args.ref, prompt_text=args.ref_text,
        prompt_lang=args.lang, text_lang=args.lang,
    )
    print("Server reachable:", tts.wait_until_ready())
    print("Wrote", tts.say(args.text, out_path=args.out))
