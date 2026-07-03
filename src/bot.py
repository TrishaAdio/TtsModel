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
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tts_client import VoiceTTS  # noqa: E402

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
TTS_BASE_URL = os.getenv("TTS_BASE_URL", "http://127.0.0.1:9880")
TEXT_LANG = os.getenv("TEXT_LANG", "en")
PROMPT_LANG = os.getenv("PROMPT_LANG", TEXT_LANG)

# Reference clip + its transcript (GPT-SoVITS needs these at inference time).
# If REF_AUDIO_PATH isn't set, auto-pick the first prepared segment.
REF_AUDIO_PATH = os.getenv("REF_AUDIO_PATH", "").strip()
REF_TEXT = os.getenv("REF_TEXT", "").strip()

# Optional: point the server at your fine-tuned checkpoints on startup.
GPT_WEIGHTS = os.getenv("GPT_WEIGHTS", "").strip()
SOVITS_WEIGHTS = os.getenv("SOVITS_WEIGHTS", "").strip()

# Optional: restrict usage to specific Telegram user IDs (comma-separated).
_allowed = os.getenv("ALLOWED_USER_IDS", "").strip()
ALLOWED_USER_IDS = {int(x) for x in _allowed.split(",") if x.strip().isdigit()} if _allowed else None

MAX_CHARS = int(os.getenv("MAX_CHARS", "600"))  # guardrail on very long messages


def _auto_reference() -> tuple[str, str]:
    """Best-effort default reference clip + transcript when not configured."""
    ref = REF_AUDIO_PATH
    if not ref:
        candidates = sorted(glob.glob(str(REPO / "data" / "dataset" / "**" / "*.wav"), recursive=True))
        ref = candidates[0] if candidates else ""
    text = REF_TEXT
    if not text and ref:
        # look for a sidecar transcript: seg_0007.txt next to seg_0007.wav
        side = Path(ref).with_suffix(".txt")
        if side.is_file():
            text = side.read_text(encoding="utf-8").strip()
    return ref, text


def wav_to_voice_ogg(wav_bytes: bytes) -> bytes:
    """Encode WAV bytes to OGG/Opus suitable for a Telegram voice note."""
    proc = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-i", "pipe:0",
            "-c:a", "libopus", "-b:a", "48k", "-ar", "48000", "-ac", "1",
            "-f", "ogg", "pipe:1",
        ],
        input=wav_bytes, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if proc.returncode != 0 or not proc.stdout:
        raise RuntimeError(f"ffmpeg encode failed: {proc.stderr.decode(errors='ignore')[:400]}")
    return proc.stdout


def build_tts() -> VoiceTTS:
    ref, text = _auto_reference()
    tts = VoiceTTS(
        base_url=TTS_BASE_URL,
        ref_audio_path=ref,
        prompt_text=text,
        prompt_lang=PROMPT_LANG,
        text_lang=TEXT_LANG,
    )
    if GPT_WEIGHTS and SOVITS_WEIGHTS:
        try:
            tts.set_weights(GPT_WEIGHTS, SOVITS_WEIGHTS)
            print(f"[bot] loaded weights: {Path(GPT_WEIGHTS).name} + {Path(SOVITS_WEIGHTS).name}")
        except Exception as e:  # noqa: BLE001
            print(f"[bot] WARNING: could not set weights: {e}")
    return tts


dp = Dispatcher()
_tts: VoiceTTS  # set in main()


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


@dp.message(F.text & ~F.text.startswith("/"))
async def on_text(message: Message) -> None:
    if not _allowed_user(message):
        return

    text = (message.text or "").strip()
    if not text:
        return
    if len(text) > MAX_CHARS:
        await message.reply(f"Message too long ({len(text)} chars). Keep it under {MAX_CHARS}.")
        return

    if not _tts.ref_audio_path or not _tts.prompt_text:
        await message.reply(
            "No reference voice configured. Set <code>REF_AUDIO_PATH</code> and "
            "<code>REF_TEXT</code> (or prepare a dataset first)."
        )
        return

    await message.bot.send_chat_action(message.chat.id, ChatAction.RECORD_VOICE)
    try:
        # sync client + ffmpeg run off the event loop
        wav = await asyncio.to_thread(_tts.synthesize, text)
        ogg = await asyncio.to_thread(wav_to_voice_ogg, wav)
    except Exception as e:  # noqa: BLE001
        await message.reply(f"Synthesis failed: <code>{str(e)[:300]}</code>")
        return

    await message.answer_voice(
        BufferedInputFile(ogg, filename="voice.ogg"),
    )


async def _amain() -> None:
    global _tts
    if not BOT_TOKEN:
        sys.exit("BOT_TOKEN is not set. export BOT_TOKEN=... and retry.")

    _tts = build_tts()
    ready = await asyncio.to_thread(_tts.wait_until_ready, 15, 2.0)
    if not ready:
        print(f"[bot] WARNING: TTS server not reachable at {TTS_BASE_URL} yet (will retry per-request).")
    if not _tts.ref_audio_path:
        print("[bot] WARNING: no reference clip found. Set REF_AUDIO_PATH/REF_TEXT.")
    else:
        print(f"[bot] reference clip: {_tts.ref_audio_path}")

    bot = Bot(BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    print("[bot] polling...")
    await dp.start_polling(bot)


if __name__ == "__main__":
    try:
        asyncio.run(_amain())
    except (KeyboardInterrupt, SystemExit):
        pass
