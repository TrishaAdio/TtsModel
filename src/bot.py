#!/usr/bin/env python3
"""
bot.py
------
Telegram bot that speaks in your fine-tuned voice.

Send it any text -> it synthesizes speech via the GPT-SoVITS API, converts the
audio to a Telegram voice note (OGG/Opus), and replies with it.

Minimum to run: a bot token.
    export BOT_TOKEN="123456:ABC..."
    python src/bot.py

Everything else has sensible defaults (see CONFIG below). The GPT-SoVITS API
server must be reachable (default http://127.0.0.1:9880 — see docs/SETUP_VPS.md).

Requires:  pip install aiogram>=3.0   (plus ffmpeg on PATH for OGG/Opus encode)
"""
from __future__ import annotations

import asyncio
import glob
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tts_client import VoiceTTS  # noqa: E402
from elevenlabs_client import ElevenLabsTTS  # noqa: E402

try:
    from aiogram import Bot, Dispatcher, F
    from aiogram.client.default import DefaultBotProperties
    from aiogram.enums import ChatAction, ParseMode
    from aiogram.filters import CommandStart
    from aiogram.types import BufferedInputFile, Message
except ImportError:
    sys.exit("Missing aiogram. Install with:  pip install 'aiogram>=3.0'")

# --------------------------------------------------------------------------- #
# CONFIG  (only BOT_TOKEN is required; the rest have defaults)
# --------------------------------------------------------------------------- #
REPO = Path(__file__).resolve().parent.parent

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()

# Which TTS engine to use: "elevenlabs" (hosted, high quality) or "gptsovits"
# (self-hosted). Default elevenlabs if an API key is present, else gptsovits.
TTS_BACKEND = os.getenv(
    "TTS_BACKEND",
    "elevenlabs" if os.getenv("ELEVENLABS_API_KEY") else "gptsovits",
).strip().lower()

# --- ElevenLabs backend config ---
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "").strip()
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "").strip()
ELEVENLABS_MODEL = os.getenv("ELEVENLABS_MODEL", "eleven_multilingual_v2").strip()

# --- GPT-SoVITS backend config ---
TTS_BASE_URL = os.getenv("TTS_BASE_URL", "http://127.0.0.1:9880")
TEXT_LANG = os.getenv("TEXT_LANG", "en")
PROMPT_LANG = os.getenv("PROMPT_LANG", TEXT_LANG)

# Sampling / segmentation knobs (env-overridable for quick tuning).
# TEXT_SPLIT controls how the API segments text before synthesis.
#   cut0 = no split (can DROP lines on multi-line/long text)
#   cut2 = split every ~50 chars  -> reliable full coverage, still smooth (default)
#   cut1 = split every 4 sentences,  cut3/cut4/cut5 = split on punctuation
TEXT_SPLIT = os.getenv("TEXT_SPLIT", "cut2").strip()
TEMPERATURE = float(os.getenv("TEMPERATURE", "1.0"))
TOP_K = int(os.getenv("TOP_K", "15"))
TOP_P = float(os.getenv("TOP_P", "1.0"))
SPEED = float(os.getenv("SPEED", "1.0"))

# Output polish: ffmpeg filter chain applied to the generated audio before it
# becomes a voice note. Removes low rumble, denoises hiss/air, and normalizes
# loudness -> crisper, "crystal clear" result. Toggle with AUDIO_CLEANUP=0.
AUDIO_CLEANUP = os.getenv("AUDIO_CLEANUP", "0") not in ("0", "false", "False", "no")
AUDIO_FILTERS = os.getenv(
    "AUDIO_FILTERS",
    "highpass=f=85,afftdn=nf=-25,loudnorm=I=-16:TP=-1.5:LRA=11",
)
OPUS_BITRATE = os.getenv("OPUS_BITRATE", "64k")

# Reference clip + its transcript (GPT-SoVITS needs these at inference time).
# If REF_AUDIO_PATH isn't set, auto-pick the LONGEST prepared segment (a good
# ~10s reference) -- never the first/shortest, which can make the model emit
# EOS immediately and return near-silent audio.
REF_AUDIO_PATH = os.getenv("REF_AUDIO_PATH", "").strip()
REF_TEXT = os.getenv("REF_TEXT", "").strip()
# Where to look up a clip's transcript (ASR .list). Auto-searched if unset.
REF_LIST = os.getenv("REF_LIST", "").strip()

# Optional: point the server at your fine-tuned checkpoints on startup.
# If unset, auto-discover the newest speaker1 weights under GSV_DIR.
GSV_DIR = os.path.expanduser(os.getenv("GSV_DIR", "~/GPT-SoVITS"))
GPT_WEIGHTS = os.getenv("GPT_WEIGHTS", "").strip()
SOVITS_WEIGHTS = os.getenv("SOVITS_WEIGHTS", "").strip()

# Optional: restrict usage to specific Telegram user IDs (comma-separated).
_allowed = os.getenv("ALLOWED_USER_IDS", "").strip()
ALLOWED_USER_IDS = {int(x) for x in _allowed.split(",") if x.strip().isdigit()} if _allowed else None

MAX_CHARS = int(os.getenv("MAX_CHARS", "600"))  # guardrail on very long messages


def _lookup_transcript(basename: str) -> str:
    """Find a clip's transcript by basename in an ASR .list file."""
    lists = []
    if REF_LIST:
        lists.append(REF_LIST)
    lists += glob.glob(os.path.join(GSV_DIR, "output", "asr_opt", "*.list"))
    lists += glob.glob(str(REPO / "**" / "*.list"), recursive=True)
    for lp in lists:
        try:
            with open(lp, encoding="utf-8") as f:
                for line in f:
                    parts = line.rstrip("\n").split("|")
                    if len(parts) >= 4 and os.path.basename(parts[0]) == basename:
                        return parts[3].strip()
        except OSError:
            continue
    return ""


def _auto_reference() -> tuple[str, str]:
    """Best-effort default reference clip + transcript when not configured.

    Picks the LONGEST prepared clip (largest file == longest, since prep writes
    a uniform 32kHz mono format capped at 10s) rather than the first one, which
    is often too short and causes immediate-EOS / near-silent output.
    """
    ref = REF_AUDIO_PATH
    if not ref:
        candidates = glob.glob(str(REPO / "data" / "dataset" / "**" / "*.wav"), recursive=True)
        ref = max(candidates, key=os.path.getsize) if candidates else ""
    text = REF_TEXT
    if not text and ref:
        # 1) sidecar transcript next to the wav (seg_0007.txt), else 2) ASR .list
        side = Path(ref).with_suffix(".txt")
        text = side.read_text(encoding="utf-8").strip() if side.is_file() else _lookup_transcript(os.path.basename(ref))
    return ref, text


def _discover_weights() -> tuple[str, str]:
    """Auto-find newest fine-tuned weights under GSV_DIR when not set via env."""
    gpt = GPT_WEIGHTS
    sovits = SOVITS_WEIGHTS
    if not gpt:
        cks = glob.glob(os.path.join(GSV_DIR, "GPT_weights*", "*.ckpt"))
        gpt = max(cks, key=os.path.getmtime) if cks else ""
    if not sovits:
        pth = glob.glob(os.path.join(GSV_DIR, "SoVITS_weights*", "*.pth"))
        sovits = max(pth, key=os.path.getmtime) if pth else ""
    return gpt, sovits


def wav_to_voice_ogg(wav_bytes: bytes) -> bytes:
    """Encode WAV bytes to OGG/Opus suitable for a Telegram voice note."""
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0"]
    if AUDIO_CLEANUP and AUDIO_FILTERS:
        cmd += ["-af", AUDIO_FILTERS]
    cmd += ["-c:a", "libopus", "-b:a", OPUS_BITRATE, "-ar", "48000", "-ac", "1", "-f", "ogg", "pipe:1"]
    proc = subprocess.run(cmd, input=wav_bytes, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0 or not proc.stdout:
        raise RuntimeError(f"ffmpeg encode failed: {proc.stderr.decode(errors='ignore')[:400]}")
    return proc.stdout


def build_synth():
    """Return the active TTS backend (must expose .synthesize(text)->bytes,
    .ready()->bool, .describe()->str)."""
    if TTS_BACKEND == "elevenlabs":
        synth = ElevenLabsTTS(
            api_key=ELEVENLABS_API_KEY,
            voice_id=ELEVENLABS_VOICE_ID,
            model_id=ELEVENLABS_MODEL,
        )
        if not synth.ready():
            print("[bot] ERROR: TTS_BACKEND=elevenlabs but ELEVENLABS_API_KEY / "
                  "ELEVENLABS_VOICE_ID are not set.")
        else:
            print(f"[bot] backend: {synth.describe()}")
        return synth
    return build_gptsovits()


def build_gptsovits() -> VoiceTTS:
    ref, text = _auto_reference()
    tts = VoiceTTS(
        base_url=TTS_BASE_URL,
        ref_audio_path=ref,
        prompt_text=text,
        prompt_lang=PROMPT_LANG,
        text_lang=TEXT_LANG,
        text_split_method=TEXT_SPLIT,
        temperature=TEMPERATURE,
        top_k=TOP_K,
        top_p=TOP_P,
        speed_factor=SPEED,
    )
    gpt_w, sovits_w = _discover_weights()
    if gpt_w and sovits_w:
        try:
            tts.set_weights(gpt_w, sovits_w)
            print(f"[bot] loaded weights: {Path(gpt_w).name} + {Path(sovits_w).name}")
        except Exception as e:  # noqa: BLE001
            print(f"[bot] WARNING: could not set weights ({e}); using the server's current voice")
    else:
        print("[bot] no fine-tuned weights found/set; using the server's current voice")
    return tts


dp = Dispatcher()
_synth = None  # active backend, set in main()


def _synth_ready(s) -> bool:
    if hasattr(s, "ready"):
        return s.ready()
    return bool(getattr(s, "ref_audio_path", "") and getattr(s, "prompt_text", ""))


def _allowed_user(message: Message) -> bool:
    if ALLOWED_USER_IDS is None:
        return True
    return bool(message.from_user and message.from_user.id in ALLOWED_USER_IDS)


@dp.message(CommandStart())
async def on_start(message: Message) -> None:
    if not _allowed_user(message):
        return
    await message.answer(
        "<b>Voice bot online.</b>\nSend me text and I'll reply with a voice note in the trained voice."
    )


def _normalize_text(raw: str) -> str:
    """Turn line breaks into sentence boundaries and collapse whitespace, so the
    TTS splitter covers every line (cut0 on multi-line text tends to drop lines)."""
    t = raw.strip()
    # blank lines / newlines -> sentence break, so each line gets synthesized
    t = re.sub(r"[ \t]*\n[ \t\n]*", ". ", t)
    t = re.sub(r"[ \t]+", " ", t)
    # avoid ".." pileups from lines that already ended in punctuation
    t = re.sub(r"([.!?])\.\s", r"\1 ", t)
    return t.strip()


@dp.message(F.text & ~F.text.startswith("/"))
async def on_text(message: Message) -> None:
    if not _allowed_user(message):
        return

    text = _normalize_text(message.text or "")
    if not text:
        return
    if len(text) > MAX_CHARS:
        await message.reply(f"Message too long ({len(text)} chars). Keep it under {MAX_CHARS}.")
        return

    if not _synth_ready(_synth):
        if TTS_BACKEND == "elevenlabs":
            await message.reply(
                "ElevenLabs not configured. Set <code>ELEVENLABS_API_KEY</code> and "
                "<code>ELEVENLABS_VOICE_ID</code>."
            )
        else:
            await message.reply(
                "No reference voice configured. Set <code>REF_AUDIO_PATH</code> and "
                "<code>REF_TEXT</code> (or prepare a dataset first)."
            )
        return

    await message.bot.send_chat_action(message.chat.id, ChatAction.RECORD_VOICE)
    try:
        # synthesize (wav for GPT-SoVITS, mp3 for ElevenLabs) then encode to a
        # Telegram voice note; ffmpeg auto-detects the input format.
        audio = await asyncio.to_thread(_synth.synthesize, text)
        ogg = await asyncio.to_thread(wav_to_voice_ogg, audio)
    except Exception as e:  # noqa: BLE001
        await message.reply(f"Synthesis failed: <code>{str(e)[:300]}</code>")
        return

    await message.answer_voice(
        BufferedInputFile(ogg, filename="voice.ogg"),
    )


async def _amain() -> None:
    global _synth
    if not BOT_TOKEN:
        sys.exit("BOT_TOKEN is not set. export BOT_TOKEN=... and retry.")

    print(f"[bot] TTS backend: {TTS_BACKEND}")
    _synth = build_synth()

    if TTS_BACKEND == "gptsovits":
        reachable = await asyncio.to_thread(_synth.wait_until_ready, 15, 2.0)
        if not reachable:
            print(f"[bot] WARNING: TTS server not reachable at {TTS_BASE_URL} yet (will retry per-request).")
        if getattr(_synth, "ref_audio_path", ""):
            print(f"[bot] reference clip: {_synth.ref_audio_path}")
        else:
            print("[bot] WARNING: no reference clip found. Set REF_AUDIO_PATH/REF_TEXT.")
    elif not _synth_ready(_synth):
        print("[bot] WARNING: ElevenLabs not configured (ELEVENLABS_API_KEY / ELEVENLABS_VOICE_ID).")

    bot = Bot(BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    print("[bot] polling...")
    await dp.start_polling(bot)


if __name__ == "__main__":
    try:
        asyncio.run(_amain())
    except (KeyboardInterrupt, SystemExit):
        pass
