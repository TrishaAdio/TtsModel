# voice-model-training

Fine-tune a natural-sounding voice model (GPT-SoVITS) from your own voice clips
and serve it over an API your AI can call. Colored CLI via `colorama`.
Optimized for a Linux GPU VPS — including Blackwell / RTX 50-series.

## Why GPT-SoVITS
Purpose-built for high-similarity, natural cloning from small data. With a
clean ~20–30 min dataset you get output that closely matches the source speaker
and avoids the "robotic" quality of lightweight TTS.

## Layout
```
voice-model-training/
├── voices.txt              # source clip URLs (one per line)
├── requirements.txt        # CLI/pipeline deps (colorama, requests, pydub, ...)
├── src/
│   ├── console.py          # colorama output helpers
│   ├── download_dataset.py # fetch clips from voices.txt -> data/raw/
│   ├── prepare_dataset.py  # slice + clean -> data/dataset/speaker1/
│   ├── tts_client.py       # call the trained voice from your AI
│   └── bot.py              # Telegram bot: text in -> voice note out (api_v2)
├── train.sh                # headless training (no WebUI / no open ports)
├── docs/
│   ├── SETUP_VPS.md        # install GPT-SoVITS + correct PyTorch for your GPU
│   ├── TRAINING.md         # WebUI training walkthrough
│   └── HEADLESS.md         # CLI-only training + Hinglish/English notes
└── data/                   # raw/ and dataset/ (gitignored)
```

## Quick start
```bash
pip install -r requirements.txt
sudo apt-get install -y ffmpeg

# 1) download the source clips
python src/download_dataset.py

# 2) slice + clean into training segments
python src/prepare_dataset.py --input-dir data/raw --output data/dataset/speaker1

# 3) on the GPU VPS: install + train  (docs/SETUP_VPS.md, docs/TRAINING.md)
# 4) serve + integrate                (src/tts_client.py)
```

## Best quality: clean retrain (one command)
If the voice sounds noisy/rough, the cause is almost always the **source audio**.
Denoise it and retrain in a single step:
```bash
cd ~/TtsModel && git pull && source venv/bin/activate
./retrain_clean.sh          # denoise -> re-slice -> retrain (fine-tuned, clean)
```
It denoises `data/raw` (neural FRCRN if available, else `noisereduce`), re-slices,
and trains a fine-tuned model on the cleaned audio. At the end it prints the exact
`export ... && python src/bot.py` command to run the bot on the new weights.
The single biggest quality lever is still cleaner/longer source recordings.

## No open ports? Train headless
If you can't use the WebUI (ports closed), run the whole thing from the CLI:
```bash
cp train.sh ~/GPT-SoVITS/ && cd ~/GPT-SoVITS && source .venv/bin/activate
DATASET_DIR=~/TtsModel/data/dataset/speaker1 LANG=en ./train.sh
```
See `docs/HEADLESS.md` for the full walkthrough.

## Language: Hinglish / English
GPT-SoVITS v2 has no Hindi phonemizer, so run Hinglish through the **English**
pipeline: `LANG=en` for training, `TEXT_LANG=en` for the bot, and type Hinglish
in **Latin script**. The clone still reproduces the Hindi sounds acoustically
from your clips. Details in `docs/HEADLESS.md`.

## Telegram bot
After the model is fine-tuned and the GPT-SoVITS API is running, run a bot that
turns any text you send into a voice note in the trained voice:
```bash
pip install "aiogram>=3.0"      # and ensure ffmpeg is installed
export BOT_TOKEN="123456:ABC..."   # the ONLY required setting
python src/bot.py
```
With just `BOT_TOKEN`, the bot auto-selects the **longest** prepared clip as the
reference (a good ~10s prompt), auto-loads its transcript from the ASR `.list`,
and auto-discovers the newest fine-tuned weights under `~/GPT-SoVITS`. Override
any of it if you want:
```bash
# export TTS_BASE_URL="http://127.0.0.1:9880"
# export GSV_DIR="$HOME/GPT-SoVITS"                 # where weights + .list live
# export REF_AUDIO_PATH="data/dataset/speaker1/seg_0048.wav"
# export REF_TEXT="exact transcript of that clip"
# export GPT_WEIGHTS="/path/speaker1-e12.ckpt"
# export SOVITS_WEIGHTS="/path/speaker1_e12_s....pth"
# export ALLOWED_USER_IDS="11111111,22222222"       # restrict who can use it
```
Send the bot text -> it replies with a voice note (OGG/Opus) spoken in your
cloned voice.

## Pipeline
1. **Download** — `download_dataset.py` pulls every URL in `voices.txt`.
2. **Prepare** — `prepare_dataset.py` converts to mono 32 kHz and slices on
   silence into 3–10s utterances.
3. **Train** — GPT-SoVITS WebUI: ASR-label → format → train SoVITS + GPT.
4. **Serve** — `api_v2.py` + `tts_client.py` wired into your AI.

> Only clone voices you own or have explicit permission to use, and disclose
> synthetic speech where appropriate.
