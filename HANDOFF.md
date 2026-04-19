# Orange Pi 5 × PowerInfer — Handoff Document
**Date:** 2026-04-19
**Status:** Model export complete. Pending: transfer to Orange Pi 5 + service deploy + validation.

---

## What Was Accomplished

### 1. rkllm_enhanced binary (DONE — deployed on Pi)
Drop-in replacement for `/usr/bin/rkllm` that uses the RKLLM SDK directly with:
- **n_keep=4** — attention-sink sliding window KV cache (constant RAM regardless of context length)
- **embed_flash=1** — NVMe embedding cache
- **enabled_cpus_mask=0xF0** — pins inference to A76 big cores (CPU4-7)
- **rkllm_set_chat_template** — overrides to U+FF5C format after init

Source: `rkllm_enhanced.cpp`
Compiled on Pi: `/usr/local/bin/rkllm_enhanced`

### 2. server_patched.py (DONE — deployed on Pi)
Six patches to `server.py` for Qwen3-4B ChatML format + zombie recovery:

| Patch | What | Why |
|-------|------|-----|
| STRIP_RE | removes `<\|im_end\|>` and `<\|im_start\|>[a-z]*\n?` from output | ChatML tokens leak into response |
| RESPONSE_MARKERS | adds `</think>` | ChatML thinking-end marker |
| END_MARKERS | adds `<\|im_end\|>` | ChatML response-end marker |
| wait_nl | treats `<think>` same as `""` / `":"` | first line triggers skip_think phase |
| zombie recovery | `OSError` → `needs_restart=True` + `rkllm_idle.set()` | rkllm_enhanced crash → auto-recover |
| RKLLM_BIN env | reads `RKLLM_BIN` env var | lets systemd override which binary to run |

Source: `server_patched.py` → deployed to `/home/ubuntu/rkllm_api/server.py`

### 3. Systemd service (DONE — on Pi)
`/etc/systemd/system/rkllm-api.service`:
```ini
Environment="RKLLM_BIN=/usr/local/bin/rkllm_enhanced"
Environment="RKLLM_MODEL_PATH=/srv/nvme-share/models/Qwen3-4B-w8a8-hybrid.rkllm"
Environment="RKLLM_MODEL_NAME=qwen3:4b"
```
> **NOTE:** Service currently points to old model path `Qwen3-4B-1.2.0.rkllm`.
> Must update `RKLLM_MODEL_PATH` after copying new models.

### 4. Qwen3-4B W8A8 models exported (DONE — on Windows, not yet on Pi)
Built with **rkllm-toolkit 1.2.1b1** in WSL2 Ubuntu:

| File | hybrid_rate | Layers | Size |
|------|-------------|--------|------|
| `Qwen3-4B-w8a8-npu.rkllm` | 0.0 | all NPU | 4.6 GB |
| `Qwen3-4B-w8a8-hybrid.rkllm` | 0.5 | 50% CPU A76 / 50% NPU | 4.6 GB |

**IMPORTANT quantization finding:**
rkllm-toolkit 1.2.1b1 does **NOT** support W4A16 for RK3588.
Tested all variants: `w4a16`, `w4a16_g32/64/128` with `grq`/`normal` — all fail with
`ERROR: target_platform: rk3588 not support quantized_dtype: w4a16`.
Only `w8a8` with `normal` algorithm works for RK3588 in this toolkit version.

Models are located at: `C:\Users\kamyar\Documents\PowerInfer\models\`

---

## Remaining Tasks

### Step 1 — Transfer models to Orange Pi 5
The Pi is on the local network. Models need to go to `/srv/nvme-share/models/`.

```bash
# From Windows (find Pi's IP first with: arp -a)
scp "C:\Users\kamyar\Documents\PowerInfer\models\Qwen3-4B-w8a8-hybrid.rkllm" ubuntu@<PI_IP>:/srv/nvme-share/models/
scp "C:\Users\kamyar\Documents\PowerInfer\models\Qwen3-4B-w8a8-npu.rkllm" ubuntu@<PI_IP>:/srv/nvme-share/models/
```

Or from Pi (pull from Windows share if SMB is enabled).

### Step 2 — Update systemd service on Pi
SSH into Pi and update the model path:
```bash
sudo nano /etc/systemd/system/rkllm-api.service
# Change RKLLM_MODEL_PATH to point to new model:
# Environment="RKLLM_MODEL_PATH=/srv/nvme-share/models/Qwen3-4B-w8a8-hybrid.rkllm"

sudo systemctl daemon-reload
sudo systemctl restart rkllm-api
sudo systemctl status rkllm-api
journalctl -u rkllm-api -f
```

### Step 3 — Validate inference
```bash
# Test API endpoint (from any machine on network)
curl -s http://<PI_IP>:8080/api/generate \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3:4b","prompt":"What is 2+2?","stream":false}' | python3 -m json.tool

# Test chain-of-thought (should produce <think>...</think> then answer)
curl -s http://<PI_IP>:8080/api/generate \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3:4b","prompt":"If a train travels 120km in 2 hours, what is its speed?","stream":false}' | python3 -m json.tool

# Or via Ollama-compatible endpoint:
curl -s http://<PI_IP>:11434/api/generate \
  -d '{"model":"qwen3:4b","prompt":"Hello","stream":false}'
```

### Step 4 — Validate features
| Feature | How to verify |
|---------|--------------|
| KV cache sliding window | Long multi-turn conversation — RAM stays flat |
| CPU+NPU hybrid | `htop` during inference — see CPU4-7 active + NPU load |
| Chain-of-thought | Ask reasoning question — response has `<think>...</think>` |
| Zombie recovery | Kill rkllm_enhanced manually, next request should auto-restart |

---

## Environment Details

### WSL2 (Windows side — model conversion)
- Distro: Ubuntu 24.04 (WSL2)
- Python venv: `/opt/rkllm-env/` (Python 3.12)
- rkllm-toolkit wheel: `convert/rkllm_toolkit-1.2.1b1-cp312-cp312-linux_x86_64.whl`
- Model source: `/convert/Qwen3-4B/` (Qwen/Qwen3-4B from HuggingFace, fp32)
- Export output: `/output/` in WSL2 → copied to `models/` in this repo folder

### Orange Pi 5 hardware
- SoC: RK3588 (4× A55 + 4× A76, 6 TOPS NPU)
- RAM: 16 GB LPDDR5
- Storage: NVMe via M.2 HAT → `/srv/nvme-share/`
- OS: Ubuntu 22.04 arm64

### Orange Pi 5 software stack
- RKLLM runtime: `/usr/local/lib/librkllmrt.so` (SDK 1.2.x)
- rkllm binary: `/usr/local/bin/rkllm_enhanced` (our custom binary)
- API server: `/home/ubuntu/rkllm_api/server.py` (patched)
- Service: `rkllm-api.service` (port 8080, Ollama-compatible on 11434)
- Current model (old): `/srv/nvme-share/models/Qwen3-4B-1.2.0.rkllm`

---

## Key Files in This Repo

```
PowerInfer/
├── rkllm_enhanced.cpp          # Custom rkllm binary (n_keep, embed_flash, A76 pinning)
├── server_patched.py           # Patched API server (ChatML + zombie recovery)
├── convert/
│   ├── export_qwen3.py         # Model export script (W8A8, hybrid_rate variants)
│   ├── Dockerfile              # Docker build env (not used — WSL2 used instead)
│   ├── run_export.sh           # Shell wrapper to run export in WSL2
│   └── rkllm_toolkit-*.whl    # rkllm-toolkit wheel (copied from Pi)
└── models/                     # (gitignored) exported .rkllm files
    ├── Qwen3-4B-w8a8-npu.rkllm    (4.6 GB, all-NPU)
    └── Qwen3-4B-w8a8-hybrid.rkllm (4.6 GB, 50% CPU+NPU)
```

---

## Compiling rkllm_enhanced on Pi
```bash
# On Orange Pi 5:
scp rkllm_enhanced.cpp ubuntu@<PI_IP>:~/
ssh ubuntu@<PI_IP>
g++ -O2 -o rkllm_enhanced rkllm_enhanced.cpp \
    -I/usr/local/include -L/usr/local/lib \
    -lrkllmrt -Wl,-rpath,/usr/local/lib
sudo cp rkllm_enhanced /usr/local/bin/rkllm_enhanced
sudo chmod +x /usr/local/bin/rkllm_enhanced
```

---

## Known Issues & Notes

1. **W4A16 not supported**: rkllm-toolkit 1.2.1b1 rejects all W4A16 dtypes for RK3588.
   If a newer toolkit version is available, try `w4a16_g128` + `normal` to halve model size.

2. **startup_buf shows ChatML**: The server.py startup capture shows Qwen3's native ChatML
   template (printed during `rkllm_init` before our override). This is cosmetic only —
   inference uses the overridden U+FF5C template via `rkllm_set_chat_template`.

3. **hybrid_rate=0.5 sensitivity analysis**: The hybrid model export ran a 2h47min
   per-block sensitivity analysis (36 blocks × ~4.6 min/block). This determines which
   layers go to CPU vs NPU. Block sensitivities ranged 0.0–1.675 across 36 transformer blocks.

4. **Pi IP address**: Not fixed. Last seen at `192.168.0.x` range. Check with `arp -a`
   or look for the device on router admin page. SSH user: `ubuntu`.
