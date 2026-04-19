"""
export_qwen3.py — Qwen3-4B turbo export for RK3588

Produces two variants:
  1. Qwen3-4B-w8a8-npu.rkllm      hybrid_rate=0   (all NPU,  ~4.6 GB)
  2. Qwen3-4B-w8a8-hybrid.rkllm   hybrid_rate=0.5 (CPU+NPU, ~4.6 GB)

Quantization: W8A8 (8-bit weights + 8-bit activations).
NOTE: rkllm-toolkit 1.2.1b1 does NOT support W4A16 for RK3588.
      All W4A16 variants (w4a16, w4a16_g32/64/128 with grq/normal) fail.
      W8A8 is the only supported quantization dtype for RK3588.
Platform: RK3588, num_npu_core=3 (all cores), max_context=4096
"""

import os, json
from rkllm.api import RKLLM

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_ID      = "Qwen/Qwen3-4B"   # HuggingFace model ID (public, no auth needed)
MODEL_DIR     = "/convert/Qwen3-4B"
OUTPUT_DIR    = "/output"
CALIB_FILE    = "/convert/calib.json"
TARGET        = "rk3588"
NUM_NPU_CORE  = 3
MAX_CONTEXT   = 4096
QUANT_DTYPE   = "w8a8"         # Only dtype supported by rkllm-toolkit 1.2.1b1 for RK3588
QUANT_ALGO    = "normal"       # standard algorithm
OPT_LEVEL     = 1

# ── Calibration dataset ────────────────────────────────────────────────────────
# Representative prompts — used to find optimal quantization scales.
# Covers reasoning, math, code, multilingual so all neuron patterns are sampled.
CALIB_PROMPTS = [
    "A farmer has 17 sheep. All but 9 die. How many are left?",
    "What is 25% of 360? Show your work step by step.",
    "Explain why the sky appears blue during the day.",
    "Write a Python function to check if a number is prime.",
    "A train leaves Chicago at 60 mph. Another leaves New York at 80 mph. The cities are 790 miles apart. When do they meet?",
    "Translate to French: The quick brown fox jumps over the lazy dog.",
    "What is the capital of Australia? Many people guess Sydney but that is wrong.",
    "Summarize the theory of general relativity in two sentences.",
    "If all roses are flowers and some flowers fade quickly, do all roses fade quickly?",
    "Write a SQL query to find the top 3 customers by total purchase amount.",
    "What is the derivative of x^3 + 2x^2 - 5x + 1?",
    "RK3588是新一代高端处理器，具有高算力、低功耗、超强多媒体、丰富数据接口等特点。请翻译成英文。",
    "给定以下Python代码，请改写为列表解析：squares = []; [squares.append(i**2) for i in range(10)]",
    "Solve: 2x + 3y = 12, x - y = 1. Find x and y.",
    "What are the main differences between supervised and unsupervised learning?",
    "def is_palindrome(s: str) -> bool: # complete this function",
    "Convert 72 degrees Fahrenheit to Celsius.",
    "What is the time complexity of quicksort in the average case?",
    "List 5 renewable energy sources and one advantage of each.",
    "If a car depreciates 15% per year, what is it worth after 3 years if it started at $30,000?",
]

CALIB_TMPL = "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

# ── Download model ─────────────────────────────────────────────────────────────
if not os.path.isdir(MODEL_DIR):
    print(f"Downloading {MODEL_ID} from HuggingFace...", flush=True)
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=MODEL_DIR,
        ignore_patterns=["*.msgpack", "*.h5", "flax_model*"],
    )
    print("Download complete.", flush=True)
else:
    print(f"Model already at {MODEL_DIR}", flush=True)

# ── Build calibration file ─────────────────────────────────────────────────────
if not os.path.exists(CALIB_FILE):
    print("Building calibration dataset...", flush=True)
    calib = [{"input": CALIB_TMPL.format(prompt=p), "target": ""} for p in CALIB_PROMPTS]
    with open(CALIB_FILE, "w") as f:
        json.dump(calib, f, ensure_ascii=False, indent=2)
    print(f"Calibration file written: {len(calib)} samples", flush=True)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Export helper ──────────────────────────────────────────────────────────────
def export(hybrid_rate: float, suffix: str):
    name = f"Qwen3-4B-w8a8-{suffix}.rkllm"
    out  = os.path.join(OUTPUT_DIR, name)
    if os.path.exists(out):
        print(f"Already exists, skipping: {name}", flush=True)
        return

    print(f"\n{'='*60}", flush=True)
    print(f"Exporting {name}  (hybrid_rate={hybrid_rate})", flush=True)
    print(f"{'='*60}\n", flush=True)

    llm = RKLLM()

    # Load model from local HuggingFace snapshot
    ret = llm.load_huggingface(
        model=MODEL_DIR,
        model_lora=None,
        device="cpu",
        dtype="float16",
        custom_config=None,
        load_weight=True,
    )
    if ret != 0:
        print(f"load_huggingface failed: {ret}", flush=True)
        return

    # Build quantized model
    ret = llm.build(
        do_quantization=True,
        optimization_level=OPT_LEVEL,
        quantized_dtype=QUANT_DTYPE,
        quantized_algorithm=QUANT_ALGO,
        target_platform=TARGET,
        num_npu_core=NUM_NPU_CORE,
        extra_qparams=None,
        dataset=CALIB_FILE,
        hybrid_rate=hybrid_rate,
        max_context=MAX_CONTEXT,
    )
    if ret != 0:
        print(f"build failed: {ret}", flush=True)
        return

    # Export .rkllm file
    ret = llm.export_rkllm(out)
    if ret != 0:
        print(f"export_rkllm failed: {ret}", flush=True)
        return

    size_gb = os.path.getsize(out) / 1024**3
    print(f"\nExported: {name}  ({size_gb:.2f} GB)", flush=True)

# ── Run both exports ───────────────────────────────────────────────────────────
# 1. Full-NPU model: fastest inference, all layers on RK3588 NPU
export(hybrid_rate=0.0, suffix="npu")

# 2. Hybrid CPU+NPU model: 50% layers on CPU A76, 50% on NPU
#    Reduces NPU memory pressure; CPU and NPU run concurrently on their layers
export(hybrid_rate=0.5, suffix="hybrid")

print("\nAll exports done. Files in /output:")
for f in os.listdir(OUTPUT_DIR):
    sz = os.path.getsize(os.path.join(OUTPUT_DIR, f)) / 1024**3
    print(f"  {f}  ({sz:.2f} GB)")
