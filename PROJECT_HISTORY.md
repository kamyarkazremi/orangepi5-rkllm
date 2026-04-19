# Orange Pi 5 × PowerInfer — Full Project History & Strategy
**Last updated:** 2026-04-19
**GitHub:** https://github.com/kamyarkazremi/orangepi5-rkllm

---

## Project Goal

Deploy a full-featured, production-grade LLM inference stack on an **Orange Pi 5** (RK3588 SoC) using Rockchip's RKLLM SDK. The stack must include:

1. **Turbo quantization** — W4A16 (attempted) / W8A8 (achieved) to minimize model size and maximize throughput
2. **Constant RAM usage** — KV cache sliding window that never grows regardless of conversation length
3. **CPU/NPU layer split** — hybrid inference using both the A76 big cores and the 6-TOPS NPU concurrently
4. **Chain-of-thought** — Qwen3-4B thinking mode (`<think>...</think>` tags) working end-to-end
5. **Service reliability** — zombie process auto-recovery, proper systemd supervision

---

## Hardware Specs

| Component | Details |
|-----------|---------|
| Board | Orange Pi 5 |
| SoC | Rockchip RK3588 |
| CPU cores | 4× Cortex-A55 (little, CPU0-3) + 4× Cortex-A76 (big, CPU4-7) |
| NPU | RKNPU2, 6 TOPS, 3 cores |
| RAM | 16 GB LPDDR5 |
| Storage | NVMe M.2 via HAT → `/srv/nvme-share/` |
| OS | Ubuntu 22.04 arm64 |

---

## Phase 1 — Understanding the Stack (Sessions 1–2)

### What existed on arrival
- `/usr/bin/rkllm` — Rockchip's stock CLI binary for RKLLM models
- `/home/ubuntu/rkllm_api/server.py` — Python API server wrapping rkllm via PTY
- `/srv/nvme-share/models/Qwen3-4B-1.2.0.rkllm` — a pre-built Qwen3-4B model (format unknown)
- `rkllm-api.service` — systemd service running the API on port 8080

### Problems found
- **No KV sliding window** — default rkllm has unbounded KV cache → RAM grows with context
- **No CPU/NPU split** — all layers on NPU only
- **No chat template control** — server hardcoded U+FF5C format; Qwen3-4B uses ChatML
- **Zombie crash** — if rkllm_enhanced dies, server.py reader thread exits silently, `rkllm_idle` never sets, all requests hang for 120s before giving up

### Strategy decided
Rather than patching the C source of rkllm, write a **drop-in replacement** (`rkllm_enhanced`) that uses the RKLLM C SDK directly and exposes all advanced parameters. Then patch server.py minimally to handle ChatML and zombie recovery.

---

## Phase 2 — rkllm_enhanced Binary (Sessions 2–3)

### File: `rkllm_enhanced.cpp`

A custom C++ binary that:

1. Links directly against `librkllmrt.so` (RKLLM runtime SDK)
2. Sets advanced `RKLLMParam` fields before `rkllm_init()`:
   - `p.n_keep = 4` — attention-sink sliding window: keeps first 4 KV slots (attention sinks), evicts older tokens when cache fills → **constant RAM**
   - `p.extend_param.embed_flash = 1` — enables embedding cache on NVMe storage
   - `p.extend_param.enabled_cpus_num = 4` and `enabled_cpus_mask = 0xF0u` — pins inference threads to CPU4-7 (A76 big cores only, 2.4 GHz)
3. After `rkllm_init()`, calls `rkllm_set_chat_template()` to override with U+FF5C format:
   ```
   <｜User｜>  →  <｜Assistant｜>
   ```
4. Main inference loop: `keep_history=0` on first turn, `keep_history=1` on all subsequent turns
5. SIGINT handler calls `rkllm_abort()` and reprints the prompt

### Key bug fixed during development
The hex escape `"\x9cAssistant"` caused a compile error because C merged `\x9c` and `A` into the multi-character hex `\x9cA` (out of char range). Fix: split the string literal:
```cpp
"<\xef\xbd\x9c" "Assistant\xef\xbd\x9c>"
```

Also needed `#include <cstddef>` before `#include <rkllm.h>` because the SDK header uses `size_t` without including it.

### Deployment
```bash
# On Orange Pi 5:
g++ -O2 -o rkllm_enhanced rkllm_enhanced.cpp \
    -I/usr/local/include -L/usr/local/lib \
    -lrkllmrt -Wl,-rpath,/usr/local/lib
sudo cp rkllm_enhanced /usr/local/bin/rkllm_enhanced
```

---

## Phase 3 — server_patched.py (Sessions 3–4)

### File: `server_patched.py` → `/home/ubuntu/rkllm_api/server.py`

The original server.py communicates with rkllm via a PTY (pseudoterminal) and parses the output through a state machine with phases: `wait_llm → wait_nl → skip_think → skip_template → collect`.

Six patches were applied:

#### Patch 1 — STRIP_RE (remove ChatML tokens from streaming output)
```python
STRIP_RE = re.compile(
    r'\x1b\[[0-9;]*m|\r|<\|im_end\|>|<\|im_start\|>[a-z]*\n?'
)
```
Without this, `<|im_end|>` and `<|im_start|>assistant\n` leaked into the API response body.

#### Patch 2 — RESPONSE_MARKERS (add `</think>` as thinking-end marker)
```python
RESPONSE_MARKERS = [
    b"<\xef\xbd\x9cEnd of turn\xef\xbd\x9c>",
    b"<\xef\xbd\x9cEnd of the conversation\xef\xbd\x9c>",
    b"<|Start of the response|>",
    b"</think>",   # ChatML: signals end of thinking block
]
```

#### Patch 3 — END_MARKERS (add `<|im_end|>` as response-end marker)
```python
END_MARKERS = [
    # ...existing markers...
    b"<|im_end|>",   # ChatML: signals end of assistant turn
]
```

#### Patch 4 — wait_nl phase: treat `<think>` as skip_think trigger
```python
if first_line in (b"", b":", b"<think>"):
    buf   = buf[nl_idx + 1:]
    phase = 'skip_think'
```
Without this, `<think>` on the first line was passed to `collect` phase instead of `skip_think`, bypassing the thinking-strip logic entirely.

#### Patch 5 — Zombie process recovery
```python
except OSError:
    print("[rkllm] reader: PTY closed — rkllm_enhanced died", flush=True)
    global needs_restart
    needs_restart = True
    rkllm_idle.set()   # unblocks rkllm_idle.wait() in run_chat
    break
```
Without this, a crashed rkllm_enhanced would leave `rkllm_idle` permanently unset, causing all subsequent requests to hang for 120s before timing out.

#### Patch 6 — RKLLM_BIN from environment
```python
RKLLM_BIN = os.environ.get("RKLLM_BIN", "/usr/bin/rkllm")
```
Allows systemd to override the binary path without editing the source file.

---

## Phase 4 — Systemd Service Update (Session 4)

### File: `/etc/systemd/system/rkllm-api.service`

Added environment variables:
```ini
[Service]
Environment="RKLLM_BIN=/usr/local/bin/rkllm_enhanced"
Environment="RKLLM_MODEL_PATH=/srv/nvme-share/models/Qwen3-4B-w8a8-hybrid.rkllm"
Environment="RKLLM_MODEL_NAME=qwen3:4b"
```

> **NOTE:** The `RKLLM_MODEL_PATH` still points to the old `Qwen3-4B-1.2.0.rkllm` — this must be updated after the new W8A8 models are transferred to the Pi.

---

## Phase 5 — Model Export Pipeline (Sessions 5–6)

### Background: Why re-export?

The pre-existing `Qwen3-4B-1.2.0.rkllm` was built by someone else with unknown settings. To get `hybrid_rate=0.5` (CPU+NPU layer split), the model must be re-exported from HuggingFace weights using rkllm-toolkit. The `hybrid_rate` parameter is baked into the `.rkllm` file — it cannot be set at runtime.

### Environment Setup

#### Problem: rkllm-toolkit is Linux x86_64 only
The Orange Pi 5 is arm64. The toolkit wheel `rkllm_toolkit-1.2.1b1-cp312-cp312-linux_x86_64.whl` only runs on x86_64 Linux. Options:
- Docker Desktop on Windows → failed (daemon not starting reliably)
- WSL2 Ubuntu → chosen solution

#### WSL2 Setup Process
```bash
wsl --install Ubuntu --no-launch   # install Ubuntu 24.04 distro
wsl -d Ubuntu -u root -- python3 -m venv /opt/rkllm-env
/opt/rkllm-env/bin/pip install torch==2.3.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu
/opt/rkllm-env/bin/pip install torchvision==0.18.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu
/opt/rkllm-env/bin/pip install \
    /mnt/c/Users/kamyar/Documents/PowerInfer/convert/rkllm_toolkit-1.2.1b1-cp312-cp312-linux_x86_64.whl
/opt/rkllm-env/bin/pip install setuptools==69.5.1   # provides pkg_resources
/opt/rkllm-env/bin/pip install 'pyarrow<15.0.0'      # datasets compat
```

Issues encountered:
- `auto_gptq` needs torch before install — fixed by installing torch first
- `numpy` version conflict — fixed by using venv instead of system packages
- `pyarrow.PyExtensionType` AttributeError — fixed by pinning pyarrow<15
- `pkg_resources` not found — setuptools 82+ dropped it; fixed by downgrading to 69.5.1
- torchvision/torch version mismatch (`torchvision::nms` missing) — fixed by pinning torchvision==0.18.0

### Critical Finding: W4A16 Not Supported for RK3588

Tested all quantization dtype combinations for `target_platform="rk3588"`:

| dtype | algorithm | result |
|-------|-----------|--------|
| w4a16_g128 | grq | FAIL |
| w4a16_g128 | normal | FAIL |
| w4a16_g64 | grq | FAIL |
| w4a16_g32 | grq | FAIL |
| w4a16 | grq | FAIL |
| **w8a8** | **normal** | **OK** |

Error message: `ERROR: target_platform: rk3588 not support quantized_dtype: w4a16`

This is hardcoded in the C extension `rkllm_base.cpython-312-x86_64-linux-gnu.so`. The binary does contain W4A16 strings and W4A16 support exists in the code — but it's gated off for RK3588. RK3576 may support it. This limitation is in toolkit version 1.2.1b1 specifically.

**Implication:** W8A8 produces 4.6 GB models (vs ~2.3 GB for W4A16). The NVMe drive has sufficient space.

### File: `convert/export_qwen3.py`

```python
MODEL_ID     = "Qwen/Qwen3-4B"
MODEL_DIR    = "/convert/Qwen3-4B"      # WSL2 filesystem (not Windows)
OUTPUT_DIR   = "/output"
QUANT_DTYPE  = "w8a8"
QUANT_ALGO   = "normal"
TARGET       = "rk3588"                  # lowercase required
NUM_NPU_CORE = 3                         # use all 3 NPU cores
MAX_CONTEXT  = 4096
```

Two exports:
1. `hybrid_rate=0.0` → `Qwen3-4B-w8a8-npu.rkllm` — all layers on NPU (fastest throughput)
2. `hybrid_rate=0.5` → `Qwen3-4B-w8a8-hybrid.rkllm` — 50% layers on CPU A76 + 50% on NPU (concurrent execution, lower NPU memory pressure)

### Export Timeline

| Step | Time |
|------|------|
| Download Qwen3-4B (3 shards, 7.6 GB) | ~8 minutes |
| Build quantized graph (547 layers) | ~25 seconds |
| Load calibration dataset (20 samples) | instant |
| Optimize with calibration (36 iterations, ~18s each) | ~11 minutes |
| Export .rkllm file | ~1 minute |
| **Per model total** | **~22 minutes** |
| Hybrid model sensitivity analysis (36 blocks × 4.6 min) | **2h47min** |
| **Grand total (both models)** | **~3.5 hours** |

The hybrid model's long sensitivity analysis calculates per-block quantization sensitivity to decide which layers go to CPU vs NPU. Block sensitivities ranged 0.0–1.675; high-sensitivity blocks go to higher-precision (W8A8) while low-sensitivity blocks go to compressed format (W8A8_G512). Result: 126 blocks W8A8 + 126 blocks W8A8_G512.

### Output Files

| File | Size | hybrid_rate | Layers |
|------|------|-------------|--------|
| Qwen3-4B-w8a8-npu.rkllm | 4.51 GB | 0.0 | 100% NPU |
| Qwen3-4B-w8a8-hybrid.rkllm | 4.54 GB | 0.5 | 50% CPU A76 + 50% NPU |

Models are in WSL2 `/output/` **and** copied to `C:\Users\kamyar\Documents\PowerInfer\models\` on Windows.

---

## Phase 6 — GitHub Push (Session 6)

### Repo: https://github.com/kamyarkazremi/orangepi5-rkllm

All code committed in one commit:
```
4e87273  Add Orange Pi 5 RKLLM deployment stack + Qwen3-4B export pipeline
```

Files committed (model files excluded — too large at 4.5+ GB each):
```
HANDOFF.md
convert/Dockerfile
convert/check_dtypes.py
convert/export_qwen3.py
convert/inspect_so.sh
convert/run_export.sh
convert/run_test_grq.sh
convert/test_grq.py
examples/rkllm/CMakeLists.txt
examples/rkllm/main.cpp
ggml-rkllm.c
ggml-rkllm.h
rkllm-npu2/hot_cold.h
rkllm_enhanced.cpp
scripts/split_rkllm.py
server_patched.py
.gitignore  (updated: added *.rkllm, models/, convert/output/, convert/*.whl)
```

---

## What's Left — Remaining Tasks

### Task 1: Transfer Models to Orange Pi 5 ⚠️ MUST DO FIRST

The new W8A8 models must be SCPed to the Pi's NVMe drive.

```bash
# From Windows PowerShell or CMD (find Pi IP with 'arp -a' or router admin):
scp "C:\Users\kamyar\Documents\PowerInfer\models\Qwen3-4B-w8a8-hybrid.rkllm" ^
    ubuntu@<PI_IP>:/srv/nvme-share/models/

scp "C:\Users\kamyar\Documents\PowerInfer\models\Qwen3-4B-w8a8-npu.rkllm" ^
    ubuntu@<PI_IP>:/srv/nvme-share/models/
```

Or from the Pi itself if Windows file sharing is enabled:
```bash
# On Pi:
scp kamyar@<WINDOWS_IP>:"/Users/kamyar/Documents/PowerInfer/models/*.rkllm" \
    /srv/nvme-share/models/
```

Expected transfer time: ~10–15 minutes per file at 100 Mbps LAN.

### Task 2: Update Systemd Service on Pi

```bash
ssh ubuntu@<PI_IP>
sudo nano /etc/systemd/system/rkllm-api.service
```

Change:
```ini
# OLD:
Environment="RKLLM_MODEL_PATH=/srv/nvme-share/models/Qwen3-4B-1.2.0.rkllm"

# NEW (use hybrid for CPU+NPU split):
Environment="RKLLM_MODEL_PATH=/srv/nvme-share/models/Qwen3-4B-w8a8-hybrid.rkllm"
```

Then:
```bash
sudo systemctl daemon-reload
sudo systemctl restart rkllm-api
sudo systemctl status rkllm-api
journalctl -u rkllm-api -f   # watch logs
```

### Task 3: Validate All Features

#### 3a — Basic inference
```bash
curl -s http://<PI_IP>:8080/api/generate \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3:4b","prompt":"What is 2+2?","stream":false}' \
  | python3 -m json.tool
```

Expected: JSON response with `"response": "4"` (or similar).

#### 3b — Chain-of-thought (thinking mode)
```bash
curl -s http://<PI_IP>:8080/api/generate \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3:4b","prompt":"If a train travels 120km in 2 hours, what is its average speed?","stream":false}' \
  | python3 -m json.tool
```

Expected: Response should NOT contain raw `<think>...</think>` tags (server strips them). The answer should be `60 km/h`.

To see raw thinking output (debugging):
```bash
# SSH to Pi and watch server logs:
journalctl -u rkllm-api -f
```

#### 3c — KV cache / constant RAM (sliding window)
```bash
# Start a long multi-turn conversation and monitor RAM:
watch -n2 'free -h'

# In another terminal, send many messages:
for i in $(seq 1 20); do
  curl -s http://<PI_IP>:8080/api/chat \
    -d "{\"model\":\"qwen3:4b\",\"messages\":[{\"role\":\"user\",\"content\":\"Turn $i: tell me a fact about the number $i\"}]}"
done
```

Expected: RSS memory stays flat (doesn't grow with turns) — proving `n_keep=4` sliding window is active.

#### 3d — CPU+NPU hybrid utilization
```bash
# On Pi while inference is running:
htop   # look for CPU4-7 active (A76 big cores), not CPU0-3
# Also:
cat /sys/kernel/debug/rknpu/load  # NPU utilization %
```

Expected: Both CPU4-7 AND NPU show utilization during inference (not just one or the other).

#### 3e — Zombie recovery
```bash
# While server is idle, manually kill rkllm_enhanced:
sudo pkill rkllm_enhanced

# Immediately send a request:
curl -s http://<PI_IP>:8080/api/generate \
  -d '{"model":"qwen3:4b","prompt":"Hello","stream":false}'
```

Expected: Request succeeds after ~5-10s restart delay (not hung for 120s).

### Task 4: Optional Optimizations

#### Try npu-only model for speed comparison
```bash
# Update service temporarily to npu model:
Environment="RKLLM_MODEL_PATH=/srv/nvme-share/models/Qwen3-4B-w8a8-npu.rkllm"
```
Compare tokens/second between npu-only and hybrid. Hybrid trades throughput for lower NPU memory pressure.

#### Try newer rkllm-toolkit for W4A16
If a newer toolkit version (post 1.2.1b1) adds W4A16 support for RK3588, the export process is already scripted in `convert/export_qwen3.py`. Just update `QUANT_DTYPE`:
```python
QUANT_DTYPE = "w4a16_g128"
QUANT_ALGO  = "normal"
```
This would halve the model file to ~2.3 GB and potentially improve inference speed.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│ Client (any machine)                                            │
│   curl http://pi-ip:8080/api/generate                          │
└────────────────────────┬────────────────────────────────────────┘
                         │ HTTP (Ollama-compatible)
┌────────────────────────▼────────────────────────────────────────┐
│ Orange Pi 5                                                      │
│                                                                  │
│  ┌─────────────────────────────────────────────┐                │
│  │ rkllm-api.service (systemd)                 │                │
│  │  server.py (Python, port 8080)              │                │
│  │  - receives HTTP requests                   │                │
│  │  - manages PTY to rkllm_enhanced            │                │
│  │  - parses ChatML output                     │                │
│  │  - streams tokens back to client            │                │
│  └───────────────────┬─────────────────────────┘                │
│                      │ PTY (pseudoterminal)                      │
│  ┌───────────────────▼─────────────────────────┐                │
│  │ rkllm_enhanced (compiled binary)            │                │
│  │  - n_keep=4 (sliding window KV cache)       │                │
│  │  - enabled_cpus_mask=0xF0 (A76 CPU4-7)     │                │
│  │  - embed_flash=1 (NVMe embedding cache)     │                │
│  └───────────────────┬─────────────────────────┘                │
│                      │ RKLLM C SDK calls                         │
│  ┌───────────────────▼─────────────────────────┐                │
│  │ librkllmrt.so (RKLLM runtime)               │                │
│  └──────────┬────────────────┬─────────────────┘                │
│             │                │                                   │
│  ┌──────────▼───┐  ┌─────────▼──────────────────┐               │
│  │ RKNPU2       │  │ CPU (A76 cores 4-7)         │               │
│  │ 3 cores      │  │ hybrid_rate=0.5 layers      │               │
│  │ 6 TOPS       │  │                             │               │
│  └──────────────┘  └─────────────────────────────┘               │
│                                                                  │
│  Model: /srv/nvme-share/models/Qwen3-4B-w8a8-hybrid.rkllm       │
│         (NVMe → PCIe → RK3588 memory bus)                       │
└──────────────────────────────────────────────────────────────────┘
```

---

## File Map

```
C:\Users\kamyar\Documents\PowerInfer\          (git repo root)
│
├── rkllm_enhanced.cpp          Custom rkllm binary (SDK direct, n_keep, embed_flash)
├── server_patched.py           Patched API server (ChatML + zombie recovery)
├── HANDOFF.md                  Quick handoff for next session
├── PROJECT_HISTORY.md          This file — full history + strategy
│
├── convert/
│   ├── export_qwen3.py         Model export script (W8A8, hybrid_rate variants)
│   ├── run_export.sh           Shell wrapper to run export in WSL2
│   ├── Dockerfile              Docker build env (not used — WSL2 used instead)
│   ├── test_grq.py             Dtype compatibility tester (found W4A16 unsupported)
│   ├── check_dtypes.py         Alternative dtype probe script
│   └── inspect_so.sh           Inspects rkllm_base .so for supported dtype strings
│
├── models/                     (gitignored — too large)
│   ├── Qwen3-4B-w8a8-npu.rkllm       4.51 GB, all-NPU
│   └── Qwen3-4B-w8a8-hybrid.rkllm   4.54 GB, 50% CPU + 50% NPU
│
├── examples/rkllm/             RKLLM example code
├── ggml-rkllm.c / .h           GGML integration layer for RKLLM
├── rkllm-npu2/                 NPU2-specific headers
└── scripts/split_rkllm.py      Utility: split rkllm model files
```

---

## WSL2 Environment (model conversion machine)

```
WSL2 distro:    Ubuntu 24.04
Python venv:    /opt/rkllm-env/  (Python 3.12.3, x86_64)
Toolkit wheel:  convert/rkllm_toolkit-1.2.1b1-cp312-cp312-linux_x86_64.whl
Model source:   /convert/Qwen3-4B/  (7.6 GB, 3 safetensor shards)
Output:         /output/Qwen3-4B-w8a8-{npu,hybrid}.rkllm
Windows copy:   C:\Users\kamyar\Documents\PowerInfer\models\
```

To activate venv and run exports again:
```bash
wsl -d Ubuntu -u root
source /opt/rkllm-env/bin/activate
python3 /mnt/c/Users/kamyar/Documents/PowerInfer/convert/export_qwen3.py
```

---

## Known Issues & Caveats

1. **W4A16 unsupported in rkllm-toolkit 1.2.1b1 for RK3588**
   All W4A16 variants fail. Only W8A8 + normal works. May be fixed in future toolkit versions.

2. **startup_buf shows ChatML format**
   rkllm_enhanced calls `rkllm_init()` first (which prints ChatML template to stdout), then overrides with `rkllm_set_chat_template()`. The server.py captures the startup output and sees ChatML markers. The actual inference uses the overridden U+FF5C template. This is cosmetic only — does not affect inference.

3. **Pi IP not static**
   Last seen in `192.168.0.x` range. Use `arp -a` from Windows or check router DHCP table. SSH user: `ubuntu`.

4. **NVMe space**
   Two W8A8 models = 9.1 GB. Check available space: `df -h /srv/nvme-share/` on Pi before transfer.

5. **rkllm_enhanced not automatically rebuilt**
   If `librkllmrt.so` is updated (SDK upgrade), rkllm_enhanced must be recompiled. It links at runtime via `Wl,-rpath` so ABI changes will cause a crash.
