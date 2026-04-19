"""
Upload Qwen3-4B W8A8 RKLLM models to HuggingFace Hub.
Run with: HF_TOKEN=hf_xxx python3 hf_upload.py
"""
import os, sys
from huggingface_hub import HfApi, create_repo

TOKEN = os.environ.get("HF_TOKEN")
if not TOKEN:
    print("ERROR: Set HF_TOKEN environment variable")
    print("  Get token at: https://huggingface.co/settings/tokens")
    print("  Then run: HF_TOKEN=hf_xxx python3 hf_upload.py")
    sys.exit(1)

api = HfApi(token=TOKEN)
user = api.whoami()["name"]
REPO_ID = f"{user}/Qwen3-4B-W8A8-RK3588"
MODEL_DIR = "/mnt/c/Users/kamyar/Documents/PowerInfer/models"

print(f"Logged in as: {user}")
print(f"Creating repo: {REPO_ID}")

create_repo(REPO_ID, repo_type="model", exist_ok=True, private=False, token=TOKEN)

files = [
    ("Qwen3-4B-w8a8-npu.rkllm",    "Qwen3-4B-w8a8-npu.rkllm"),
    ("Qwen3-4B-w8a8-hybrid.rkllm", "Qwen3-4B-w8a8-hybrid.rkllm"),
]

for local_name, repo_name in files:
    path = f"{MODEL_DIR}/{local_name}"
    size_gb = os.path.getsize(path) / 1024**3
    print(f"\nUploading {local_name} ({size_gb:.2f} GB)...")
    api.upload_file(
        path_or_fileobj=path,
        path_in_repo=repo_name,
        repo_id=REPO_ID,
        repo_type="model",
        token=TOKEN,
    )
    print(f"  -> done")

# Upload README card
card = f"""---
license: apache-2.0
tags:
  - rkllm
  - rk3588
  - orange-pi-5
  - qwen3
  - quantized
  - w8a8
---

# Qwen3-4B W8A8 for RK3588 (Orange Pi 5)

Quantized [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) models for Rockchip RK3588 NPU using [RKLLM](https://github.com/airockchip/rknn-llm).

## Files

| File | hybrid_rate | Description | Size |
|------|-------------|-------------|------|
| `Qwen3-4B-w8a8-npu.rkllm` | 0.0 | All layers on NPU — fastest throughput | 4.51 GB |
| `Qwen3-4B-w8a8-hybrid.rkllm` | 0.5 | 50% CPU A76 + 50% NPU — lower NPU memory pressure | 4.54 GB |

## Quantization Details

- **Source model:** Qwen/Qwen3-4B (HuggingFace)
- **Toolkit:** rkllm-toolkit 1.2.1b1
- **dtype:** W8A8 (8-bit weights + 8-bit activations)
- **Algorithm:** normal
- **Platform:** rk3588, num_npu_core=3
- **Max context:** 4096 tokens
- **Calibration:** 20 representative prompts (reasoning, math, code, multilingual)

> **Note:** W4A16 is not supported by rkllm-toolkit 1.2.1b1 for RK3588.

## Usage

See the full deployment stack at:
https://github.com/kamyarkazremi/orangepi5-rkllm

Includes:
- `rkllm_enhanced` binary with n_keep=4 sliding-window KV cache
- Patched API server with ChatML support and zombie recovery
- Systemd service configuration
"""

import tempfile, os
with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
    f.write(card)
    tmp = f.name

api.upload_file(
    path_or_fileobj=tmp,
    path_in_repo="README.md",
    repo_id=REPO_ID,
    repo_type="model",
    token=TOKEN,
)
os.unlink(tmp)

print(f"\nAll done! View at: https://huggingface.co/{REPO_ID}")
