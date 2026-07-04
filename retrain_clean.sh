#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# retrain_clean.sh - ONE command: denoise -> re-slice -> retrain the voice.
#
# Produces a fine-tuned model trained on de-noised audio, so the clone sounds
# like the dataset speaker without the background hiss/air that made earlier
# runs noisy/robotic.
#
# Run from the TtsModel repo, with the venv active:
#   cd ~/TtsModel && source venv/bin/activate && ./retrain_clean.sh
#
# Overrides (env):
#   EXP_NAME (default speaker1clean)  SOVITS_EPOCHS(10)  GPT_EPOCHS(20)  SAVE_EVERY(5)
#   TTS_DIR (~/TtsModel)  GSV_DIR (~/GPT-SoVITS)  DENOISE_METHOD (auto|frcrn|noisereduce)
# ---------------------------------------------------------------------------
set -euo pipefail

TTS_DIR="${TTS_DIR:-$HOME/TtsModel}"
GSV_DIR="${GSV_DIR:-$HOME/GPT-SoVITS}"
EXP_NAME="${EXP_NAME:-speaker1clean}"
SOVITS_EPOCHS="${SOVITS_EPOCHS:-10}"
GPT_EPOCHS="${GPT_EPOCHS:-20}"
SAVE_EVERY="${SAVE_EVERY:-5}"
DENOISE_METHOD="${DENOISE_METHOD:-auto}"

echo "=============================================================="
echo " Clean retrain:  denoise -> slice -> train"
echo "   exp:     $EXP_NAME"
echo "   epochs:  SoVITS=$SOVITS_EPOCHS  GPT=$GPT_EPOCHS  (save every $SAVE_EVERY)"
echo "   denoise: $DENOISE_METHOD"
echo "=============================================================="

cd "$TTS_DIR"

# make sure a denoiser is available (noisereduce always; FRCRN best-effort)
pip install -q noisereduce addict simplejson sortedcontainers >/dev/null 2>&1 || true

echo ""
echo ">>> [1/3] Denoise raw clips  (data/raw -> data/raw_clean)"
test -d data/raw || { echo "data/raw missing - run: python src/download_dataset.py"; exit 1; }
python src/denoise_dataset.py --input data/raw --output data/raw_clean --method "$DENOISE_METHOD"

echo ""
echo ">>> [2/3] Slice + prepare cleaned audio  (-> data/dataset/$EXP_NAME)"
python src/prepare_dataset.py --input-dir data/raw_clean --output "data/dataset/$EXP_NAME"

echo ""
echo ">>> [3/3] Train on the cleaned dataset"
cp "$TTS_DIR/train.sh" "$GSV_DIR/train.sh"
cd "$GSV_DIR"
DATASET_DIR="$TTS_DIR/data/dataset/$EXP_NAME" LANG=en EXP_NAME="$EXP_NAME" \
  ASR_DEVICE=cpu \
  SOVITS_EPOCHS="$SOVITS_EPOCHS" GPT_EPOCHS="$GPT_EPOCHS" SAVE_EVERY="$SAVE_EVERY" \
  ./train.sh

echo ""
echo "=============================================================="
echo " DONE. Fine-tuned (clean) weights:"
GPTW=$(ls -1 "$GSV_DIR"/GPT_weights_v2/${EXP_NAME}-*.ckpt 2>/dev/null | sort -V | tail -1 || true)
SOVW=$(ls -1 "$GSV_DIR"/SoVITS_weights_v2/${EXP_NAME}_*.pth 2>/dev/null | sort -V | tail -1 || true)
echo "   GPT    = ${GPTW:-<none found>}"
echo "   SoVITS = ${SOVW:-<none found>}"
echo ""
echo " Start the bot with these (matched fine-tuned pair, output-cleanup OFF):"
echo "   export GPT_WEIGHTS=\"$GPTW\""
echo "   export SOVITS_WEIGHTS=\"$SOVW\""
echo "   export REF_AUDIO_PATH=\$(ls -S $TTS_DIR/data/dataset/$EXP_NAME/*.wav | head -1)"
echo "   export REF_TEXT=\"\$(grep -m1 \"\$(basename \$REF_AUDIO_PATH)|\" $GSV_DIR/output/asr_opt/$EXP_NAME.list | cut -d'|' -f4)\""
echo "   export AUDIO_CLEANUP=0"
echo "   cd $TTS_DIR && python src/bot.py"
echo "=============================================================="
