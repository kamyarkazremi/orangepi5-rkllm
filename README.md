# Orange Pi 5 × RKLLM — Qwen3-4B Deployment Stack

Deploy **Qwen3-4B** with full hardware acceleration on **Orange Pi 5** (Rockchip RK3588):
sliding-window KV cache, CPU+NPU hybrid inference, chain-of-thought, and Ollama-compatible HTTP API.

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

---

## Features

| Feature | How | Status |
|---------|-----|--------|
| **Constant RAM** | `n_keep=4` sliding-window KV cache (attention sinks) | ✅ |
| **CPU+NPU hybrid** | `hybrid_rate=0.5` splits layers between A76 big cores and RKNPU2 | ✅ |
| **Chain-of-thought** | Qwen3-4B thinking mode (`<think>…</think>`) parsed and streamed | ✅ |
| **NVMe embedding cache** | `embed_flash=1` offloads embedding table to NVMe | ✅ |
| **Big-core pinning** | `enabled_cpus_mask=0xF0` locks threads to A76 CPU4-7 (2.4 GHz) | ✅ |
| **Zombie recovery** | Auto-restarts `rkllm_enhanced` on crash without restarting the API service | ✅ |
| **Ollama-compatible API** | Drop-in HTTP API on port 8080/11434 | ✅ |

---

## Hardware

Tested on **Orange Pi 5** with Rockchip **RK3588** SoC:

| Component | Spec |
|-----------|------|
| CPU | 4× Cortex-A55 (1.8 GHz) + 4× Cortex-A76 (2.4 GHz) |
| NPU | RKNPU2 — 6 TOPS, 3 cores |
| RAM | 16 GB LPDDR5 |
| Storage | NVMe M.2 via HAT |
| OS | Ubuntu 22.04 arm64 |

---

## Quick Start

### 1. Clone and get the models

```bash
git clone https://github.com/kamyarkazremi/orangepi5-rkllm
cd orangepi5-rkllm
```

Download the pre-built W8A8 RKLLM models from HuggingFace and place them on your NVMe drive:

```bash
# On Orange Pi 5:
pip install huggingface-hub
huggingface-cli download kamyarkazremi/Qwen3-4B-W8A8-RK3588 \
    --local-dir /srv/nvme-share/models/
```

Or download individual files:
- `Qwen3-4B-w8a8-hybrid.rkllm` — **recommended** (CPU+NPU, 4.54 GB)
- `Qwen3-4B-w8a8-npu.rkllm` — all-NPU fastest (4.51 GB)

### 2. Build and install rkllm_enhanced

```bash
# On Orange Pi 5:
g++ -O2 -o rkllm_enhanced rkllm_enhanced.cpp \
    -I/usr/local/include -L/usr/local/lib \
    -lrkllmrt -Wl,-rpath,/usr/local/lib

sudo cp rkllm_enhanced /usr/local/bin/rkllm_enhanced
sudo chmod +x /usr/local/bin/rkllm_enhanced
```

### 3. Deploy the API server

```bash
sudo cp server_patched.py /home/ubuntu/rkllm_api/server.py
```

### 4. Configure and start the service

```bash
sudo tee /etc/systemd/system/rkllm-api.service > /dev/null << 'EOF'
[Unit]
Description=RKLLM API Server
After=network.target

[Service]
User=ubuntu
Environment="RKLLM_BIN=/usr/local/bin/rkllm_enhanced"
Environment="RKLLM_MODEL_PATH=/srv/nvme-share/models/Qwen3-4B-w8a8-hybrid.rkllm"
Environment="RKLLM_MODEL_NAME=qwen3:4b"
ExecStart=/usr/bin/python3 /home/ubuntu/rkllm_api/server.py
Restart=on-failure

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now rkllm-api
```

### 5. Test the API

```bash
# Basic inference
curl http://localhost:8080/api/generate \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3:4b","prompt":"What is the capital of France?","stream":false}'

# Chain-of-thought reasoning
curl http://localhost:8080/api/generate \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3:4b","prompt":"If a train travels 120km in 2 hours, what is its speed?","stream":false}'
```

---

## Architecture

```
Client (HTTP)
     │
     ▼
server_patched.py  ←  port 8080 (Ollama-compatible)
     │  PTY
     ▼
rkllm_enhanced     ←  n_keep=4, embed_flash=1, cpus_mask=0xF0
     │  RKLLM SDK
     ▼
librkllmrt.so
   ├── RKNPU2 (50% of layers)   ← hybrid_rate=0.5
   └── A76 CPU4-7 (50% layers)  ← enabled_cpus_mask=0xF0
         │
         ▼
   Qwen3-4B-w8a8-hybrid.rkllm  (NVMe)
```

---

## Components

### `rkllm_enhanced.cpp`

Drop-in replacement for `/usr/bin/rkllm` that directly uses the RKLLM C SDK with all advanced parameters exposed:

- **`n_keep=4`** — attention-sink sliding window KV cache. Keeps the first 4 tokens (attention sinks) permanently, evicts older tokens when cache fills. RAM usage stays constant regardless of conversation length.
- **`embed_flash=1`** — stores the embedding table on NVMe instead of DRAM, saving ~200 MB RAM.
- **`enabled_cpus_mask=0xF0`** — pins all inference threads to CPU4-7 (Cortex-A76 big cores at 2.4 GHz), avoiding the slower A55 little cores.
- **`rkllm_set_chat_template`** — overrides the chat template after init to use the U+FF5C delimiter format compatible with the API server.
- SIGINT handling via `rkllm_abort()` for clean mid-generation stops.

### `server_patched.py`

Patched version of the RKLLM API server with six improvements over the original:

1. **STRIP_RE** — strips `<|im_end|>` and `<|im_start|>` ChatML tokens from streamed output
2. **RESPONSE_MARKERS** — adds `</think>` as a thinking-end marker (Qwen3-4B thinking mode)
3. **END_MARKERS** — adds `<|im_end|>` as a response-end marker
4. **wait_nl phase** — correctly routes `<think>` first-line to `skip_think` phase instead of `collect`
5. **Zombie recovery** — `OSError` in reader thread sets `needs_restart=True` and unblocks the idle event, allowing automatic recovery when `rkllm_enhanced` crashes
6. **`RKLLM_BIN` env var** — binary path configurable via environment without editing source

### `convert/export_qwen3.py`

Model conversion pipeline for building `.rkllm` files from HuggingFace weights on an x86_64 Linux machine (or WSL2 on Windows):

```python
# Produces two variants:
export(hybrid_rate=0.0, suffix="npu")     # all-NPU, fastest
export(hybrid_rate=0.5, suffix="hybrid")  # 50% CPU A76 + 50% NPU
```

**Requirements:** `rkllm-toolkit 1.2.1b1`, Python 3.12, torch 2.3.0 (CPU), x86_64 Linux or WSL2.

> **Note:** `rkllm-toolkit 1.2.1b1` does not support W4A16 for RK3588. Only W8A8 is available for this platform in this toolkit version.

---

## Models

Pre-built W8A8 models available on HuggingFace:

**[kamyarkazremi/Qwen3-4B-W8A8-RK3588](https://huggingface.co/kamyarkazremi/Qwen3-4B-W8A8-RK3588)**

| File | Size | Use case |
|------|------|----------|
| `Qwen3-4B-w8a8-hybrid.rkllm` | 4.54 GB | Recommended — CPU+NPU split reduces NPU memory pressure |
| `Qwen3-4B-w8a8-npu.rkllm` | 4.51 GB | All-NPU — higher throughput if NPU memory is not a constraint |

Both models support up to 4096 token context with the sliding-window KV cache providing constant memory regardless of conversation length.

---

## Building Models (Re-export)

If you want to rebuild the models (e.g., with different `max_context` or `hybrid_rate`):

### Requirements (x86_64 Linux or WSL2)

```bash
python3 -m venv /opt/rkllm-env
/opt/rkllm-env/bin/pip install torch==2.3.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu
/opt/rkllm-env/bin/pip install torchvision==0.18.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu
/opt/rkllm-env/bin/pip install rkllm_toolkit-1.2.1b1-cp312-cp312-linux_x86_64.whl
/opt/rkllm-env/bin/pip install setuptools==69.5.1 'pyarrow<15.0.0'
```

### Run export

```bash
/opt/rkllm-env/bin/python3 convert/export_qwen3.py
# Output: /output/Qwen3-4B-w8a8-{npu,hybrid}.rkllm
```

Export time: ~22 minutes per model (hybrid takes ~3.5 hours due to per-block sensitivity analysis).

---

## Project Status

| Task | Status |
|------|--------|
| rkllm_enhanced binary | ✅ Complete |
| server_patched.py | ✅ Complete |
| Model export pipeline | ✅ Complete |
| W8A8 models built and uploaded | ✅ Complete |
| Service deployed on Orange Pi 5 | ✅ Complete |
| Chain-of-thought validation | 🔄 Pending (Pi offline during export) |
| W4A16 quantization | ❌ Not supported by toolkit 1.2.1b1 for RK3588 |

See [HANDOFF.md](HANDOFF.md) and [PROJECT_HISTORY.md](PROJECT_HISTORY.md) for full context.

---

## License

MIT — see [LICENSE](LICENSE).

Original PowerInfer project: [SJTU-IPADS/PowerInfer](https://github.com/SJTU-IPADS/PowerInfer)
