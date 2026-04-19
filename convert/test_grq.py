"""Quick test: try each dtype+algo combo to find what rk3588 accepts."""
from rkllm.api import RKLLM

MODEL_DIR = "/convert/Qwen3-4B"

combos = [
    ("w4a16_g128", "grq"),
    ("w4a16_g128", "normal"),
    ("w4a16_g64",  "grq"),
    ("w4a16_g32",  "grq"),
    ("w4a16",      "grq"),
    ("w8a8",       "normal"),
]

# Load model once
llm = RKLLM()
ret = llm.load_huggingface(model=MODEL_DIR, model_lora=None, device="cpu",
                            dtype="float16", custom_config=None, load_weight=True)
print(f"Load: {ret}")

for dtype, algo in combos:
    # Re-init after each failed build requires reload — use a fresh object
    llm2 = RKLLM()
    llm2.load_huggingface(model=MODEL_DIR, model_lora=None, device="cpu",
                          dtype="float16", custom_config=None, load_weight=True)
    ret = llm2.build(
        do_quantization=True,
        optimization_level=1,
        quantized_dtype=dtype,
        quantized_algorithm=algo,
        target_platform="rk3588",
        num_npu_core=3,
        extra_qparams=None,
        dataset=None,
        hybrid_rate=0,
        max_context=512,
    )
    status = "OK" if ret == 0 else "FAIL"
    print(f"{status}: dtype={dtype} algo={algo}")
    if ret == 0:
        print("Found working combination — stopping.")
        break
