# Headless training (no WebUI, no open ports)

If you can't open ports / use the browser WebUI, train entirely from the command
line with `train.sh`. It runs the same pipeline the WebUI does: ASR -> feature
extraction -> SoVITS training -> GPT training.

## Language: Hinglish / English
GPT-SoVITS v2 has phonemizers for en/zh/ja/ko/yue — **not Hindi**. For Hinglish
(Hindi-English mix) the reliable approach is to run the **English** pipeline:
- ASR with `LANG=en` transcribes into Latin script (romanized).
- The English g2p handles that text.
- The clone still reproduces the real Hindi sounds because GPT-SoVITS learns
  them acoustically from your clips, not only from phonemes.

So keep `LANG=en` for training, `TEXT_LANG=en` for the bot/inference, and when
you message the bot, type Hinglish in **Latin script** (e.g. "kya haal hai, how
are you") rather than Devanagari for best results.

> If you later need true Devanagari Hindi, that requires a model/build with a
> Hindi frontend; v2 won't phonemize it.

## Steps

### 1. Prepare the dataset (in this repo)
```bash
cd ~/TtsModel && source .venv/bin/activate 2>/dev/null || true
python src/download_dataset.py
python src/prepare_dataset.py --input-dir data/raw --output data/dataset/speaker1
```

### 2. Install GPT-SoVITS + PyTorch for your GPU
Follow `docs/SETUP_VPS.md` §1–4 (clone repo, venv, correct torch wheel,
`pip install -r requirements.txt`, download pretrained models). Skip §5 (WebUI).

### 3. Run headless training
```bash
cp ~/TtsModel/train.sh ~/GPT-SoVITS/train.sh
cd ~/GPT-SoVITS && source .venv/bin/activate

DATASET_DIR=~/TtsModel/data/dataset/speaker1 \
LANG=en \
EXP_NAME=speaker1 \
SOVITS_EPOCHS=12 GPT_EPOCHS=15 \
./train.sh
```
Everything else has defaults (VERSION=v2, GPU=0, batch=4, save every 4 epochs).
With ~11 min of audio, try SoVITS 12–15 / GPT 15–20; watch for over-memorization.

Outputs:
- `SoVITS_weights_v2/speaker1*.pth`
- `GPT_weights_v2/speaker1*.ckpt`

### 4. Serve + use (still no public ports)
```bash
python api_v2.py -a 127.0.0.1 -p 9880      # bound to localhost only
```
Then from this repo, with the two weight paths:
```bash
cd ~/TtsModel
export BOT_TOKEN="123456:ABC..."
export TEXT_LANG=en
export GPT_WEIGHTS=~/GPT-SoVITS/GPT_weights_v2/speaker1-e15.ckpt
export SOVITS_WEIGHTS=~/GPT-SoVITS/SoVITS_weights_v2/speaker1_e12_s....pth
export REF_AUDIO_PATH=~/TtsModel/data/dataset/speaker1/seg_0007.wav
export REF_TEXT="exact transcript of that clip"   # in Latin script
python src/bot.py
```
The bot binds nothing public — it makes an outbound connection to Telegram, so
no inbound ports are needed. The TTS API stays on localhost.

## Troubleshooting
| Symptom | Fix |
|---|---|
| `sm_120 not compatible` | wrong torch wheel — reinstall cu128 (SETUP_VPS.md §3) |
| CUDA OOM | lower `SOVITS_BS`/`GPT_BS` to 2–3 |
| ASR list empty | check clips exist in DATASET_DIR; try `ASR_SIZE=large-v3` |
| pretrained not found | you skipped the model download in SETUP_VPS.md §4 |
| robotic / flat output | more GPT epochs; at inference use temperature ~0.95 |
