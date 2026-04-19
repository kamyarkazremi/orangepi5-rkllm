FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 python3-pip git wget curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /convert

# Install rkllm-toolkit (Rockchip RKLLM conversion SDK)
COPY rkllm_toolkit-1.2.1b1-cp312-cp312-linux_x86_64.whl .
RUN pip install --break-system-packages \
    rkllm_toolkit-1.2.1b1-cp312-cp312-linux_x86_64.whl

# Install PyTorch (CPU-only — no CUDA needed for quantization)
RUN pip install --break-system-packages \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install HuggingFace + model dependencies
RUN pip install --break-system-packages \
    transformers accelerate huggingface-hub sentencepiece protobuf

COPY export_qwen3.py .

# Output dir (mount from host)
RUN mkdir -p /output

CMD ["python3", "export_qwen3.py"]
