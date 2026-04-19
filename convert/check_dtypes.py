"""Try each dtype for rk3588 to find which ones work."""
from rkllm.api import RKLLM
import os

MODEL_DIR = "/convert/Qwen3-4B"
TARGET = "RK3588"

dtypes = ["w8a8", "w4a8_g32", "w4a16_g32", "w4a16_g64", "w4a16_g128", "w4a16", "w8a8_g128"]

for dtype in dtypes:
    llm = RKLLM()
    ret = llm.load_huggingface(model=MODEL_DIR, model_lora=None, device="cpu", dtype="float16", custom_config=None, load_weight=True)
    if ret != 0:
        print(f"{dtype}: load failed")
        continue
    ret = llm.build(
        do_quantization=True,
        optimization_level=1,
        quantized_dtype=dtype,
        quantized_algorithm="normal",
        target_platform=TARGET,
        num_npu_core=3,
        extra_qparams=None,
        dataset=None,
        hybrid_rate=0,
        max_context=512,
    )
    if ret == 0:
        print(f"SUCCESS: {dtype}")
        out = f"/output/test_{dtype}.rkllm"
        llm.export_rkllm(out)
        os.remove(out)
        break
    else:
        print(f"FAIL: {dtype}")
