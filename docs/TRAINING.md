# Training Walkthrough — clips → natural cloned voice

Steps 1–3 run from this repo (CPU is fine). Steps 4+ run in the GPT-SoVITS
WebUI on your GPU VPS. Budget: well under an hour of training on a 5070-class GPU.

---

## 1. Download the source clips
```bash
pip install -r requirements.txt
sudo apt-get install -y ffmpeg
python src/download_dataset.py            # voices.txt -> data/raw/
```

## 2. Prepare (slice + clean)
```bash
python src/prepare_dataset.py --input-dir data/raw --output data/dataset/speaker1
# add --denoise only if there's steady background hiss
```
Result: `data/dataset/speaker1/seg_0001.wav ...` (mono, 32 kHz, 3–10s each).

Dataset quality is everything. Spot-check a few clips: single speaker, no
music/overlap, clear speech, consistent volume. Dirty clips → robotic output.

## 3. Pick a reference clip for inference
Choose ONE clean 5–10s segment and note its **exact** transcript. You pass
these to `src/tts_client.py` as `ref_audio_path` + `prompt_text`.

## 4. WebUI: label the dataset
Tab **0-Fetch dataset**: point the tools at `data/dataset/speaker1`, run
**ASR** (Faster-Whisper for en/multilingual). Skim/fix the transcripts — wrong
text hurts pronunciation.

## 5. WebUI: one-click formatting
Tab **1A-Dataset formatting**: set experiment name (`speaker1`), point at the
audio dir + the ASR `.list`, run all formatting steps.

## 6. Train SoVITS (timbre / quality)
| Param        | Value  | Notes |
|--------------|--------|-------|
| Batch size   | 4–6    | lower to 2–3 on OOM (12 GB) |
| Total epochs | 8–15   | start 10 |
| fp16         | on     | faster; fine on Blackwell/Ada/Ampere |

## 7. Train GPT (prosody — the "not robotic" part)
| Param        | Value  | Notes |
|--------------|--------|-------|
| Batch size   | 4–6    | lower on OOM |
| Total epochs | 10–20  | start 15 |

## 8. Inference test (WebUI tab 1C)
Select your trained GPT + SoVITS weights, load the reference clip + transcript,
generate. Tune: **top_k 10–20**, **temperature 0.9–1.0**. Lower temp = stabler
but flatter; higher = more expressive but can wobble.

## 9. Serve to your AI
```bash
python api_v2.py -a 127.0.0.1 -p 9880
```
```python
from tts_client import VoiceTTS
tts = VoiceTTS(base_url="http://127.0.0.1:9880",
               ref_audio_path=".../seg_0007.wav", prompt_text="...",
               prompt_lang="en", text_lang="en", top_k=15, temperature=0.95)
tts.set_weights(gpt_weights_path=".../speaker1-e15.ckpt",
                sovits_weights_path=".../speaker1_e10_s....pth")
audio = tts.synthesize("This is my AI speaking in the cloned voice.")
```

---

## Troubleshooting
| Symptom | Cause | Fix |
|---|---|---|
| `sm_120 not compatible` | wrong torch wheel | reinstall cu128 (SETUP_VPS.md §3) |
| CUDA out of memory | batch too high | batch 2–3, fp16 on |
| Robotic / flat | too few GPT epochs / low temp | more GPT epochs, temp ~0.95 |
| Wrong words | bad ASR transcripts | fix the `.list`, retrain |
| Not like the person | dirty data / bad ref clip | cleaner clips, better reference |

## Consent
Only clone a voice you own or have permission to use, and disclose synthetic
speech where appropriate.
