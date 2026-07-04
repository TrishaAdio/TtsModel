#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# train.sh - fully headless GPT-SoVITS training (no WebUI, no open ports)
#
# Replicates what the GPT-SoVITS WebUI does, from the command line:
#   0) ASR transcribe your prepared clips  -> a .list file
#   1) feature extraction (text / hubert / semantic)
#   2) train the SoVITS model  (timbre / audio quality)
#   3) train the GPT model     (prosody / naturalness)
#
# Run it FROM the GPT-SoVITS repo (its venv active), pointing at the dataset
# produced by prepare_dataset.py.
#
# Usage:
#   cp ~/TtsModel/train.sh ~/GPT-SoVITS/train.sh
#   cd ~/GPT-SoVITS && source .venv/bin/activate
#   DATASET_DIR=~/TtsModel/data/dataset/speaker1 LANG=en ./train.sh
#
# Common overrides (env vars):
#   EXP_NAME       experiment name           (default speaker1)
#   DATASET_DIR    folder of prepared wavs    (REQUIRED)
#   LANG           auto|en|ja|ko              (default en)
#   VERSION        v2|v2Pro|v2ProPlus|v1      (default v2)
#   GPU            gpu index                  (default 0)
#   SOVITS_EPOCHS  (default 12)   SOVITS_BS (default 4)
#   GPT_EPOCHS     (default 15)   GPT_BS    (default 4)
#   SAVE_EVERY     save weights every N ep    (default 4)
#   ASR_SIZE       faster-whisper size        (default large-v3)
#   ASR_PREC       float16|float32|int8       (default float16)
# ---------------------------------------------------------------------------
set -euo pipefail

# ---- config ---------------------------------------------------------------
EXP_NAME="${EXP_NAME:-speaker1}"
DATASET_DIR="${DATASET_DIR:?Set DATASET_DIR=/abs/path/to/prepared/wavs}"
LANG="${LANG:-en}"
VERSION="${VERSION:-v2}"
GPU="${GPU:-0}"
SOVITS_EPOCHS="${SOVITS_EPOCHS:-12}"
SOVITS_BS="${SOVITS_BS:-4}"
GPT_EPOCHS="${GPT_EPOCHS:-15}"
GPT_BS="${GPT_BS:-4}"
SAVE_EVERY="${SAVE_EVERY:-4}"
ASR_SIZE="${ASR_SIZE:-large-v3}"
ASR_PREC="${ASR_PREC:-float16}"
ASR_DEVICE="${ASR_DEVICE:-auto}"   # auto|cpu  (use cpu if ctranslate2 lacks sm_120)
SKIP_ASR="${SKIP_ASR:-0}"          # 1 = reuse an existing .list (skip transcription)

DATASET_DIR="$(readlink -f "$DATASET_DIR")"
PY="python"
PM="GPT_SoVITS/pretrained_models"

# GPT-SoVITS subprocess scripts import packages from two roots:
#   - repo root        -> tools.*, GPT_SoVITS.*
#   - GPT_SoVITS/ dir  -> text, AR, module, feature_extractor, utils
# The WebUI puts both on sys.path via a .pth file; headless we set them here.
# This script must be run FROM the GPT-SoVITS repo root.
export PYTHONPATH="$(pwd):$(pwd)/GPT_SoVITS:${PYTHONPATH:-}"
EXP_DIR="logs/${EXP_NAME}"

# pretrained model paths for the chosen version (v2 defaults)
case "$VERSION" in
  v1)
    S2G="$PM/s2G488k.pth"
    S1="$PM/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt" ;;
  v2|v2Pro|v2ProPlus)
    S2G="$PM/gsv-v2final-pretrained/s2G2333k.pth"
    S1="$PM/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt" ;;
  *) echo "Unsupported VERSION=$VERSION"; exit 1 ;;
esac
S2D="${S2G/s2G/s2D}"
BERT="$PM/chinese-roberta-wwm-ext-large"
HUBERT="$PM/chinese-hubert-base"

# Final-weight output dirs (v1 has no suffix; v2+ are versioned)
if [ "$VERSION" = "v1" ]; then
  SOVITS_WDIR="SoVITS_weights"; GPT_WDIR="GPT_weights"
else
  SOVITS_WDIR="SoVITS_weights_$VERSION"; GPT_WDIR="GPT_weights_$VERSION"
fi

echo "=============================================================="
echo " Headless GPT-SoVITS training"
echo "   exp:        $EXP_NAME  ($VERSION)"
echo "   dataset:    $DATASET_DIR"
echo "   language:   $LANG        gpu: $GPU"
echo "   SoVITS:     bs=$SOVITS_BS epochs=$SOVITS_EPOCHS"
echo "   GPT:        bs=$GPT_BS epochs=$GPT_EPOCHS"
echo "=============================================================="

# Pre-create every output dir the trainers write to but do NOT create themselves.
# (s2_train saves to $EXP_DIR/logs_s2_$VERSION and to the weight dirs via my_save,
#  which does shutil.move without makedirs — the WebUI normally pre-creates these.)
mkdir -p "$EXP_DIR" "$EXP_DIR/logs_s2_$VERSION" "$EXP_DIR/logs_s1_$VERSION" \
         "$SOVITS_WDIR" "$GPT_WDIR" output/asr_opt tmp

# ---- 0) ASR: transcribe clips into a .list file ---------------------------
LIST="output/asr_opt/$(basename "$DATASET_DIR").list"
if [ "$SKIP_ASR" = "1" ] && [ -s "$LIST" ]; then
  echo "[0/4] ASR skipped, reusing $LIST ($(wc -l < "$LIST") lines)"
else
  echo "[0/4] ASR transcription -> $LIST  (device=$ASR_DEVICE)"
  ASR_ENV=()
  ASR_PREC_EFF="$ASR_PREC"
  if [ "$ASR_DEVICE" = "cpu" ]; then
    # force CPU (ctranslate2 on Blackwell often has no sm_120 kernels)
    ASR_ENV=(env CUDA_VISIBLE_DEVICES=)
    [ "$ASR_PREC" = "float16" ] && ASR_PREC_EFF="int8"   # CPU can't do fp16
  fi
  "${ASR_ENV[@]}" "$PY" -s tools/asr/fasterwhisper_asr.py \
    -i "$DATASET_DIR" -o output/asr_opt -s "$ASR_SIZE" -l "$LANG" -p "$ASR_PREC_EFF"
  test -s "$LIST" || { echo "ASR produced no list file"; exit 1; }
  echo "      transcribed $(wc -l < "$LIST") lines"
fi

export inp_text="$LIST"
export inp_wav_dir="$DATASET_DIR"
export exp_name="$EXP_NAME"
export opt_dir="$EXP_DIR"
export is_half="True"
export i_part="0"
export all_parts="1"
export _CUDA_VISIBLE_DEVICES="$GPU"

# ---- 1a) text + phoneme features ------------------------------------------
echo "[1/4] feature extraction: text/phoneme"
export bert_pretrained_dir="$BERT"
"$PY" -s GPT_SoVITS/prepare_datasets/1-get-text.py
cat "$EXP_DIR/2-name2text-0.txt" > "$EXP_DIR/2-name2text.txt"

# ---- 1b) self-supervised (hubert) features --------------------------------
echo "      feature extraction: hubert/ssl"
export cnhubert_base_dir="$HUBERT"
export sv_path="$PM/sv/pretrained_eres2netv2w24s4ep4.ckpt"
"$PY" -s GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py
if [[ "$VERSION" == *Pro* ]]; then
  "$PY" -s GPT_SoVITS/prepare_datasets/2-get-sv.py
fi

# ---- 1c) semantic tokens --------------------------------------------------
echo "      feature extraction: semantic tokens"
if [[ "$VERSION" == "v2Pro" || "$VERSION" == "v2ProPlus" ]]; then
  export s2config_path="GPT_SoVITS/configs/s2${VERSION}.json"
else
  export s2config_path="GPT_SoVITS/configs/s2.json"
fi
export pretrained_s2G="$S2G"
"$PY" -s GPT_SoVITS/prepare_datasets/3-get-semantic.py
{ echo -e "item_name\tsemantic_audio"; cat "$EXP_DIR/6-name2semantic-0.tsv"; } > "$EXP_DIR/6-name2semantic.tsv"

# ---- 2) train SoVITS (s2) -------------------------------------------------
echo "[2/4] training SoVITS ($SOVITS_EPOCHS epochs)"
S2CFG="GPT_SoVITS/configs/s2.json"
[[ "$VERSION" == "v2Pro" || "$VERSION" == "v2ProPlus" ]] && S2CFG="GPT_SoVITS/configs/s2${VERSION}.json"
"$PY" - "$S2CFG" tmp/tmp_s2.json <<PYEOF
import json, os, sys
src, dst = sys.argv[1], sys.argv[2]
d = json.load(open(src))
d["train"]["batch_size"] = $SOVITS_BS
d["train"]["epochs"] = $SOVITS_EPOCHS
d["train"]["save_every_epoch"] = $SAVE_EVERY
d["train"]["if_save_latest"] = True
d["train"]["if_save_every_weights"] = True
d["train"]["gpu_numbers"] = "$GPU"
d["train"]["pretrained_s2G"] = "$S2G"
d["train"]["pretrained_s2D"] = "$S2D"
d["train"]["fp16_run"] = True
d["model"]["version"] = "$VERSION"
d["data"]["exp_dir"] = d["s2_ckpt_dir"] = "$EXP_DIR"
d["save_weight_dir"] = "SoVITS_weights_$VERSION" if "$VERSION"!="v1" else "SoVITS_weights"
d["name"] = "$EXP_NAME"; d["version"] = "$VERSION"
json.dump(d, open(dst, "w"))
print("wrote", dst)
PYEOF
if [[ "$VERSION" == "v3" || "$VERSION" == "v4" ]]; then
  "$PY" -s GPT_SoVITS/s2_train_v3_lora.py --config tmp/tmp_s2.json
else
  "$PY" -s GPT_SoVITS/s2_train.py --config tmp/tmp_s2.json
fi

# ---- 3) train GPT (s1) ----------------------------------------------------
echo "[3/4] training GPT ($GPT_EPOCHS epochs)"
S1CFG="GPT_SoVITS/configs/s1longer.yaml"
[[ "$VERSION" != "v1" ]] && S1CFG="GPT_SoVITS/configs/s1longer-v2.yaml"
export _CUDA_VISIBLE_DEVICES="$GPU"
export hz="25hz"
"$PY" - "$S1CFG" tmp/tmp_s1.yaml <<PYEOF
import yaml, sys
src, dst = sys.argv[1], sys.argv[2]
d = yaml.safe_load(open(src))
d["train"]["batch_size"] = $GPT_BS
d["train"]["epochs"] = $GPT_EPOCHS
d["train"]["save_every_n_epoch"] = $SAVE_EVERY
d["train"]["if_save_every_weights"] = True
d["train"]["if_save_latest"] = True
d["train"]["if_dpo"] = False
d["train"]["precision"] = "16-mixed"
d["train"]["half_weights_save_dir"] = "GPT_weights_$VERSION" if "$VERSION"!="v1" else "GPT_weights"
d["train"]["exp_name"] = "$EXP_NAME"
d["pretrained_s1"] = "$S1"
d["train_semantic_path"] = "$EXP_DIR/6-name2semantic.tsv"
d["train_phoneme_path"] = "$EXP_DIR/2-name2text.txt"
d["output_dir"] = "$EXP_DIR/logs_s1_$VERSION"
yaml.dump(d, open(dst, "w"), default_flow_style=False)
print("wrote", dst)
PYEOF
"$PY" -s GPT_SoVITS/s1_train.py --config_file tmp/tmp_s1.yaml

# ---- done -----------------------------------------------------------------
echo "=============================================================="
echo "[4/4] DONE."
echo "  SoVITS weights -> SoVITS_weights_${VERSION}/ (look for ${EXP_NAME}*.pth)"
echo "  GPT weights    -> GPT_weights_${VERSION}/   (look for ${EXP_NAME}*.ckpt)"
echo ""
echo "Serve it:   python api_v2.py -a 127.0.0.1 -p 9880"
echo "Then point tts_client.py / bot.py at those two weight files."
echo "=============================================================="
