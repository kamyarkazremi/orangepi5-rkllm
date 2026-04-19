#!/bin/bash
strings /opt/rkllm-env/lib/python3.12/site-packages/rkllm/api/rkllm_base.cpython-312-x86_64-linux-gnu.so | grep -i "rk3588\|w4a16\|w8a8\|not support\|quantized_dtype" | head -60
