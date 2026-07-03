# GPT-SoVITS on a Linux GPU VPS — Setup

Goal: fine-tune a voice model on your clips that sounds like the real speaker
(natural, not robotic), then serve it over an API your AI calls.

---

## 0. Know your GPU first

```bash
nvidia-smi
```
Note the **GPU model** and **Driver / CUDA version**.

| GPU family                      | Arch      | Compute (sm) | PyTorch wheel |
|---------------------------------|-----------|--------------|---------------|
| RTX 50xx (5070/5080/5090)       | Blackwell | **sm_120**   | **cu128**, driver **570+** |
| RTX 40xx (4090/4080/L4)         | Ada       | sm_89        | cu121/cu124 |
| RTX 30xx (3090/3080), A100      | Ampere    | sm_80/86     | cu121/cu124 |

> #1 failure on RTX 50-series: an old PyTorch. If you see
> `sm_120 is not compatible` or `no kernel image is available`, your torch
> wheel is too old — reinstall with cu128 (step 3).

---

## 1. System packages
```bash
sudo apt-get update
sudo apt-get install -y git ffmpeg build-essential
```

## 2. Clone GPT-SoVITS + env
```bash
git clone https://github.com/RVC-Boss/GPT-SoVITS
cd GPT-SoVITS
conda create -n gptsovits python=3.10 -y && conda activate gptsovits
# or: python3.10 -m venv .venv && source .venv/bin/activate
```

## 3. Install PyTorch that MATCHES your GPU (BEFORE other deps)
Blackwell / RTX 50xx:
```bash
pip install --upgrade pip
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
# if that won't resolve for your Python:
# pip install --pre torch torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```
Ada / Ampere (40xx/30xx/A100):
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Verify the GPU actually runs kernels:
```bash
python - <<'PY'
import torch
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
x = torch.randn(1024, 1024, device="cuda"); print("matmul ok:", bool((x@x).sum().item()))
PY
```
Want `cuda: True`, your GPU name, and `matmul ok: True`. A 5070 reports `(12, 0)`.

## 4. GPT-SoVITS deps + pretrained models
```bash
pip install -r requirements.txt
bash install.sh          # if present; else follow README "Pretrained Models"
```

## 5. Launch the WebUI (tunnel from your laptop)
```bash
python webui.py
# on YOUR machine:
ssh -L 9874:localhost:9874 user@YOUR_VPS_IP   # forward whatever port it prints
```
Open `http://localhost:9874` locally.

## 6. Train  →  see docs/TRAINING.md
## 7. Serve  →  `python api_v2.py -a 127.0.0.1 -p 9880`, then use src/tts_client.py

> Keep the API bound to localhost / behind a firewall. Don't expose the raw
> port publicly.
