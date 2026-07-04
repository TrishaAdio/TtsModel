#!/usr/bin/env python3
"""
elevenlabs_client.py
--------------------
ElevenLabs text-to-speech client. Same shape as tts_client.VoiceTTS
(a .synthesize(text) -> audio bytes method) so the bot can use either backend.

Setup (2 minutes, exactly what impressed you):
  1. Create an account at https://elevenlabs.io and clone your voice in their UI
     (upload a sample -> get a Voice).
  2. Grab your API key (Profile -> API Keys) and the Voice ID.
  3. export ELEVENLABS_API_KEY="..."   ELEVENLABS_VOICE_ID="..."

Find your voice IDs from the CLI:
  python src/elevenlabs_client.py --list

Quick synth test:
  python src/elevenlabs_client.py --voice <VOICE_ID> --text "Hello from my clone" --out out.mp3
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import requests

API_ROOT = "https://api.elevenlabs.io/v1"


@dataclass
class ElevenLabsTTS:
    api_key: str
    voice_id: str
    model_id: str = "eleven_multilingual_v2"   # or "eleven_turbo_v2_5" (faster/cheaper)
    stability: float = 0.5
    similarity_boost: float = 0.85
    style: float = 0.0
    use_speaker_boost: bool = True
    output_format: str = "mp3_44100_128"
    timeout: int = 120
    session: requests.Session = field(default_factory=requests.Session, repr=False)

    def ready(self) -> bool:
        return bool(self.api_key and self.voice_id)

    def describe(self) -> str:
        return f"ElevenLabs voice={self.voice_id} model={self.model_id}"

    def synthesize(self, text: str, **_ignore) -> bytes:
        """Return MP3 audio bytes for `text`. Raises on API error."""
        if not self.ready():
            raise ValueError("ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID are required.")
        url = f"{API_ROOT}/text-to-speech/{self.voice_id}"
        headers = {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        }
        body = {
            "text": text,
            "model_id": self.model_id,
            "voice_settings": {
                "stability": self.stability,
                "similarity_boost": self.similarity_boost,
                "style": self.style,
                "use_speaker_boost": self.use_speaker_boost,
            },
        }
        resp = self.session.post(
            url, headers=headers, params={"output_format": self.output_format},
            json=body, timeout=self.timeout,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"ElevenLabs TTS failed ({resp.status_code}): {resp.text[:300]}")
        return resp.content

    # compatibility no-ops so the bot can treat both backends the same
    def wait_until_ready(self, *_a, **_k) -> bool:
        return self.ready()


def list_voices(api_key: str) -> list[dict]:
    r = requests.get(f"{API_ROOT}/voices", headers={"xi-api-key": api_key}, timeout=30)
    r.raise_for_status()
    return r.json().get("voices", [])


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="ElevenLabs TTS client / voice lister.")
    ap.add_argument("--list", action="store_true", help="List your voices (id + name).")
    ap.add_argument("--voice", default=os.getenv("ELEVENLABS_VOICE_ID", ""))
    ap.add_argument("--text", default="Hello, this is my ElevenLabs cloned voice.")
    ap.add_argument("--model", default=os.getenv("ELEVENLABS_MODEL", "eleven_multilingual_v2"))
    ap.add_argument("--out", default="out.mp3")
    args = ap.parse_args()

    key = os.getenv("ELEVENLABS_API_KEY", "").strip()
    if not key:
        raise SystemExit("Set ELEVENLABS_API_KEY first.")

    if args.list:
        for v in list_voices(key):
            print(f"{v.get('voice_id')}\t{v.get('name')}\t({v.get('category')})")
        raise SystemExit(0)

    if not args.voice:
        raise SystemExit("Pass --voice <VOICE_ID> (or set ELEVENLABS_VOICE_ID). Use --list to find it.")

    tts = ElevenLabsTTS(api_key=key, voice_id=args.voice, model_id=args.model)
    with open(args.out, "wb") as f:
        f.write(tts.synthesize(args.text))
    print("wrote", args.out)
