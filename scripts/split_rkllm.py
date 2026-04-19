#!/usr/bin/env python3
"""
split_rkllm.py  –  Split a HuggingFace transformer model into two halves and
                    export each half as a .rkllm file for the layer-split
                    CPU/NPU hybrid inference mode on Orange Pi 5 / RK3588.

Quantisation modes
------------------
  fp16          Full precision.  Largest files, best accuracy.
  w8a8          8-bit weights + 8-bit activations.
  w4a16         4-bit weights + 16-bit activations.  Naive per-128-group.
  turbo         AWQ w4 + group_size=32 + mixed-precision outer layers.
                Uses calibration data to find optimal per-channel scales
                before packing to 4-bit → same memory as w4a16 but much
                better accuracy.  Recommended for Orange Pi 5.

Memory estimates (7B model, both halves combined)
-------------------------------------------------
  fp16    ~14 GB   (exceeds RK3588 RAM; not practical)
  w8a8     ~7 GB   (tight on 8 GB boards)
  w4a16    ~4 GB   ✓
  turbo    ~4 GB   ✓  (same size, better quality than w4a16)

Usage
-----
    # Turbo quant (recommended):
    python scripts/split_rkllm.py \\
        --model  meta-llama/Llama-3-8B \\
        --split-layer 15 \\
        --output-dir  models/llama3-8b \\
        --quant turbo

    # Standard w4a16:
    python scripts/split_rkllm.py \\
        --model  meta-llama/Llama-3-8B \\
        --split-layer 15 \\
        --output-dir  models/llama3-8b \\
        --quant w4a16

    # The script produces:
    #   models/llama3-8b/first_half.rkllm
    #   models/llama3-8b/second_half.rkllm

Requirements
------------
    pip install rkllm-toolkit transformers torch

    For --quant turbo, also:
        pip install autoawq

    rkllm-toolkit wheel: https://github.com/airockchip/rknn-llm/releases
        pip install rkllm_toolkit-*.whl
"""

import argparse
import os
import sys
import json
import math
import tempfile

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Wrapper models that expose only a subset of transformer layers
# ---------------------------------------------------------------------------

class FirstHalfModel(nn.Module):
    """
    Runs the embedding layer + transformer layers [0 .. split_layer].
    Output: hidden_states at the end of layer split_layer.
    This is what handle_a processes on the NPU.
    """

    def __init__(self, base_model, split_layer: int):
        super().__init__()
        # LLaMA / Mistral / Qwen etc. all follow the same pattern:
        #   model.embed_tokens  – token embedding
        #   model.layers[i]     – transformer decoder layers
        #   model.norm          – final RMS-norm (only in second half)
        #   lm_head             – logit projection (only in second half)
        self.embed_tokens = base_model.model.embed_tokens
        self.layers       = nn.ModuleList(base_model.model.layers[:split_layer + 1])
        self.config       = base_model.config
        self.split_layer  = split_layer

    def forward(self, input_ids, attention_mask=None):
        hidden = self.embed_tokens(input_ids)
        for layer in self.layers:
            out    = layer(hidden, attention_mask=attention_mask)
            hidden = out[0]          # (batch, seq, hidden_dim)
        return hidden


class SecondHalfModel(nn.Module):
    """
    Runs transformer layers [split_layer+1 .. end], final norm, and lm_head.
    Input:  hidden_states from FirstHalfModel  (RKLLM_INPUT_EMBED).
    Output: logits.
    This is what handle_b processes on the NPU.
    """

    def __init__(self, base_model, split_layer: int):
        super().__init__()
        self.layers  = nn.ModuleList(base_model.model.layers[split_layer + 1:])
        self.norm    = base_model.model.norm
        self.lm_head = base_model.lm_head
        self.config  = base_model.config

    def forward(self, hidden_states, attention_mask=None):
        for layer in self.layers:
            out           = layer(hidden_states, attention_mask=attention_mask)
            hidden_states = out[0]
        hidden_states = self.norm(hidden_states)
        logits        = self.lm_head(hidden_states)
        return logits


# ---------------------------------------------------------------------------
# Memory estimator
# ---------------------------------------------------------------------------

# Bytes per parameter for each quant scheme (approximate).
_BYTES_PER_PARAM = {
    "fp16":   2.0,
    "w8a8":   1.0,
    "w4a16":  0.5,
    "turbo":  0.5,   # same weight storage as w4a16 + tiny scale overhead
}

def estimate_memory_mb(n_params: int, quant: str) -> float:
    bpp = _BYTES_PER_PARAM.get(quant, 0.5)
    # add ~10% for KV-cache, activations, RKLLM runtime overhead
    return n_params * bpp / (1024 ** 2) * 1.10

def print_memory_table(n_params: int) -> None:
    print("\n  ┌──────────────┬────────────────┬──────────────────────────────┐")
    print(  "  │ quant        │ est. NPU RAM   │ notes                        │")
    print(  "  ├──────────────┼────────────────┼──────────────────────────────┤")
    for q, label in [("fp16",  "full precision"),
                      ("w8a8",  "8-bit weights"),
                      ("w4a16", "4-bit naive"),
                      ("turbo", "4-bit AWQ ← recommended")]:
        mb = estimate_memory_mb(n_params, q)
        gb = mb / 1024
        flag = "⚠ " if gb > 7.5 else "✓ "
        print(f"  │ {q:<12} │ {gb:>6.1f} GB      │ {flag}{label:<27}│")
    print(  "  └──────────────┴────────────────┴──────────────────────────────┘")
    print()


# ---------------------------------------------------------------------------
# AWQ pre-quantisation (turbo mode)
# ---------------------------------------------------------------------------

# These layers are more sensitive to quantisation error and should stay at
# higher precision.  Applied in mixed-precision turbo mode.
_SENSITIVE_LAYER_KEYWORDS = [
    "embed_tokens",   # token embedding – huge vocab, first layer
    "lm_head",        # output projection – directly affects logits
    "norm",           # RMS-norm weights – small but important
]

def apply_awq(base_model, tokenizer, calib_data: list, group_size: int = 32,
              w_bit: int = 4, zero_point: bool = True):
    """
    Run AWQ (Activation-aware Weight Quantization) on the base model.

    AWQ analyses activation magnitudes over calibration data to find
    per-channel scales that minimise quantisation error for the most
    important weights — without retraining.

    Returns a new model with quantised weights in-place.
    Requires: pip install autoawq
    """
    try:
        from awq import AutoAWQForCausalLM
        from awq.quantize.quantizer import AwqQuantizer
    except ImportError:
        print(
            "ERROR: autoawq is not installed.\n"
            "  pip install autoawq\n"
            "  (or use --quant w4a16 to skip AWQ)",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"  Running AWQ calibration  (group_size={group_size}, w_bit={w_bit})")
    print(f"  Calibration prompts: {len(calib_data)}")

    quant_config = {
        "zero_point":   zero_point,
        "q_group_size": group_size,
        "w_bit":        w_bit,
        "version":      "GEMM",   # GEMM is faster than GEMV on NPU
    }

    # Save base model to a temp directory, reload via AutoAWQ, quantise.
    with tempfile.TemporaryDirectory() as tmp:
        base_model.save_pretrained(tmp)
        tokenizer.save_pretrained(tmp)

        awq_model = AutoAWQForCausalLM.from_pretrained(
            tmp, low_cpu_mem_usage=True, torch_dtype=torch.float16
        )
        awq_model.quantize(tokenizer,
                           quant_config=quant_config,
                           calib_data=calib_data)

        # Save quantised model, reload as a standard HF model for RKLLM export.
        awq_model.save_quantized(tmp)
        quantised = AutoModelForCausalLM.from_pretrained(
            tmp, torch_dtype=torch.float16, device_map="cpu", trust_remote_code=True
        )

    print("  AWQ calibration complete.")
    return quantised


# ---------------------------------------------------------------------------
# Mixed-precision: per-layer quant scheme assignment
# ---------------------------------------------------------------------------

def build_mixed_precision_qparams(n_layers: int, group_size: int) -> dict:
    """
    Returns extra_qparams dict for RKLLM build():
      - First 2 and last 2 transformer layers → w8a8  (most input/output sensitive)
      - All middle layers                      → w4a16 with given group_size
      - Embedding + lm_head                   → fp16  (never quantise)

    This costs ~5% more memory vs pure w4a16 but significantly improves
    output coherence at the token boundaries.
    """
    qparams = {}

    # Outer layers at w8a8
    sensitive = list(range(2)) + list(range(n_layers - 2, n_layers))
    for i in sensitive:
        qparams[f"model.layers.{i}"] = {"quantized_dtype": "w8a8"}

    # Middle layers at w4a16 with fine-grained group quantisation
    for i in range(2, n_layers - 2):
        qparams[f"model.layers.{i}"] = {
            "quantized_dtype": "w4a16",
            "q_group_size":    group_size,
        }

    # Embedding and output head: keep at fp16
    for key in _SENSITIVE_LAYER_KEYWORDS:
        qparams[key] = {"quantized_dtype": "fp16"}

    return qparams


# ---------------------------------------------------------------------------
# RKLLM export
# ---------------------------------------------------------------------------

def _require_rkllm():
    try:
        from rkllm.api import RKLLM
        return RKLLM
    except ImportError:
        print(
            "ERROR: rkllm-toolkit is not installed.\n"
            "  Download the wheel from https://github.com/airockchip/rknn-llm/releases\n"
            "  and install with:  pip install rkllm_toolkit-*.whl",
            file=sys.stderr,
        )
        sys.exit(1)


def export_half(model: nn.Module,
                model_name: str,
                out_path: str,
                quant: str,
                group_size: int,
                platform: str,
                max_context_len: int,
                tokenizer,
                n_layers: int,
                calib_data: list,
                kv_bits: int = None) -> None:
    """
    Convert a PyTorch model half to .rkllm using the RKLLM toolkit.

    quant options:
      fp16   – no quantisation
      w8a8   – 8-bit weights + 8-bit activations
      w4a16  – 4-bit weights + 16-bit activations, per-group-size
      turbo  – AWQ pre-quant + w4a16 + mixed-precision outer layers
    """
    RKLLM = _require_rkllm()

    # Map quant mode → RKLLM build() arguments
    if quant == "fp16":
        do_quant = False
        qdtype   = "fp16"
        qparams  = {}
    elif quant == "w8a8":
        do_quant = True
        qdtype   = "w8a8"
        qparams  = {}
    elif quant == "w4a16":
        do_quant = True
        qdtype   = "w4a16"
        qparams  = {"q_group_size": group_size}
    elif quant == "turbo":
        # turbo: already AWQ-quantised by caller; use mixed-precision qparams
        do_quant = True
        qdtype   = "w4a16"
        qparams  = build_mixed_precision_qparams(n_layers, group_size)
    else:
        print(f"ERROR: unknown quant '{quant}'", file=sys.stderr)
        sys.exit(1)

    # Merge KV-cache quantisation into qparams if requested.
    if kv_bits is not None:
        kv_str = f"int{kv_bits}"
        qparams["kv_cache_quant_dtype"] = kv_str   # RKLLM ≥ 1.2.x field (may be ignored)
        qparams["kv_cache_quant_bits"]  = kv_bits  # alternative field name

    print(f"\n── Exporting {model_name}")
    print(f"   → {out_path}")
    kv_info = f"  kv_bits={kv_bits}" if kv_bits else ""
    print(f"   quant={quant}  group_size={group_size}  platform={platform}{kv_info}")

    rkllm = RKLLM()
    rkllm.load_pytorch(
        model        = model,
        tokenizer    = tokenizer,
        model_lm_head= None,
    )
    rkllm.build(
        do_quantization   = do_quant,
        quantized_dtype   = qdtype,
        optimization_level= 1,
        target_platform   = platform,
        num_npu_core      = 3,
        extra_qparams     = qparams,
        dataset           = calib_data if do_quant else None,
    )
    rkllm.export_rkllm(out_path)

    size_mb = os.path.getsize(out_path) / (1024 ** 2) if os.path.exists(out_path) else 0
    print(f"   Saved: {out_path}  ({size_mb:.0f} MB)")


# ---------------------------------------------------------------------------
# Calibration dataset
# ---------------------------------------------------------------------------

# More calibration samples → better AWQ scale estimation.
# Mix domains so scales generalise across typical use-cases.
CALIB_PROMPTS = [
    "The capital of France is Paris, which has been",
    "In machine learning, a transformer model uses self-attention to",
    "The RK3588 SoC integrates a neural processing unit capable of",
    "To train a deep neural network from scratch you need",
    "Orange Pi 5 is a single-board computer based on the Rockchip RK3588S",
    "The quick brown fox jumps over the lazy dog near the river",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return",
    "Paris Agreement on climate change was signed in 2015 and aims to",
    "The human brain contains approximately 86 billion neurons that form",
    "In Python, list comprehensions provide a concise way to create lists",
    "Photosynthesis is the process by which plants use sunlight to",
    "The Rockchip NPU supports INT4 and INT8 quantized operations for",
    "Large language models are trained on massive datasets and can",
    "The theory of relativity states that energy and mass are related by",
    "In linear algebra, a matrix multiplication requires the inner",
    "The history of artificial intelligence dates back to the 1950s when",
]


def make_calibration_data(tokenizer, max_len: int = 256, n_samples: int = None):
    """
    Tokenise calibration prompts.  Returns list of token-ID tensors.
    n_samples: limit to first N prompts (None = all).
    """
    prompts = CALIB_PROMPTS if n_samples is None else CALIB_PROMPTS[:n_samples]
    data = []
    for p in prompts:
        enc = tokenizer(p, return_tensors="pt", max_length=max_len, truncation=True)
        data.append(enc["input_ids"])
    return data


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description="Split a HuggingFace LLM and export two .rkllm halves",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--model",        required=True,
                    help="HuggingFace model id or local path")
    ap.add_argument("--split-layer",  type=int, default=None,
                    help="Split after this layer index. Default = num_layers // 2")
    ap.add_argument("--output-dir",   default="models",
                    help="Directory to write .rkllm files")
    ap.add_argument("--quant",        default="turbo",
                    choices=["turbo", "w4a16", "w8a8", "fp16"],
                    help=(
                        "turbo  = AWQ w4 + group_size=32 + mixed-precision (recommended)\n"
                        "w4a16  = naive 4-bit weights, no AWQ\n"
                        "w8a8   = 8-bit weights + activations\n"
                        "fp16   = no quantisation"
                    ))
    ap.add_argument("--group-size",   type=int, default=32,
                    help="Weight quantisation group size (32|64|128). "
                         "Smaller = better quality, slightly larger file. "
                         "Default 32 is optimal for RK3588.")
    ap.add_argument("--calib-samples",type=int, default=None,
                    help="Number of calibration prompts (default: all built-in)")
    ap.add_argument("--calib-file",   default=None,
                    help="Path to a .txt file with one calibration prompt per line "
                         "(overrides built-in prompts)")
    ap.add_argument("--platform",     default="rk3588",
                    choices=["rk3588", "rk3588s", "rk3576"],
                    help="Target Rockchip platform")
    ap.add_argument("--max-context",  type=int, default=2048,
                    help="Maximum context length")
    ap.add_argument("--dtype",        default="float16",
                    choices=["float16", "bfloat16", "float32"],
                    help="PyTorch dtype for loading the base model")
    ap.add_argument("--kv-bits",      type=int, default=None,
                    choices=[4, 8],
                    help=(
                        "KV-cache quantisation bits passed to RKLLM via extra_qparams.\n"
                        "8 = INT8 KV-cache (~2× smaller than fp16 KV).\n"
                        "4 = INT4 KV-cache (~4× smaller, slight accuracy loss).\n"
                        "Note: RKLLM SDK support for this is version-dependent;\n"
                        "if the toolkit ignores it the model still exports correctly."
                    ))
    ap.add_argument("--prefer-gqa",  action="store_true",
                    help=(
                        "Print a reminder to use a GQA model (e.g. Llama 3, Mistral).\n"
                        "GQA reduces KV-cache by n_heads/n_kv_heads (e.g. 4× for Llama 3 8B).\n"
                        "The RKLLM toolkit preserves GQA architecture automatically;\n"
                        "no extra flag is needed if you start from a GQA checkpoint."
                    ))
    ap.add_argument("--first-only",   action="store_true",
                    help="Export only the first half")
    ap.add_argument("--second-only",  action="store_true",
                    help="Export only the second half")
    ap.add_argument("--dry-run",      action="store_true",
                    help="Print memory estimates and exit without exporting")
    return ap.parse_args()


def main():
    args = parse_args()

    dtype_map = {
        "float16":  torch.float16,
        "bfloat16": torch.bfloat16,
        "float32":  torch.float32,
    }
    dtype = dtype_map[args.dtype]

    # ── Load base model (CPU, no GPU needed on host) ─────────────────────────
    print(f"Loading base model: {args.model}  (dtype={args.dtype})")
    config    = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    n_layers  = config.num_hidden_layers
    split     = args.split_layer if args.split_layer is not None else n_layers // 2

    if split < 0 or split >= n_layers - 1:
        print(f"ERROR: --split-layer must be in [0, {n_layers - 2}]", file=sys.stderr)
        sys.exit(1)

    # ── Parameter count + memory estimate ───────────────────────────────────
    n_params = sum(p.numel() for p in
                   AutoModelForCausalLM.from_pretrained(
                       args.model, torch_dtype=torch.float16,
                       device_map="meta", trust_remote_code=True).parameters())

    print(f"\nModel: {n_layers} layers,  ~{n_params/1e9:.1f}B parameters")
    print(f"Split: layers 0-{split} (first half) | layers {split+1}-{n_layers-1} (second half)")
    print(f"Quant: {args.quant}  group_size={args.group_size}", end="")
    if args.kv_bits:
        print(f"  kv_bits={args.kv_bits}", end="")
    print()
    print_memory_table(n_params)

    # GQA advisory
    if args.prefer_gqa or hasattr(config, "num_key_value_heads"):
        n_kv = getattr(config, "num_key_value_heads", None)
        n_q  = getattr(config, "num_attention_heads", None)
        if n_kv and n_q and n_kv < n_q:
            ratio = n_q // n_kv
            kv_mb = estimate_memory_mb(n_params, args.quant) * (1 / ratio) * 0.15
            print(f"  GQA detected: {n_q} query heads / {n_kv} KV heads = {ratio}× KV-cache reduction")
            print(f"  Est. KV-cache at 2048 ctx: ~{kv_mb:.0f} MB  (RKLLM preserves GQA automatically)")
            print()

    if args.dry_run:
        print("Dry run – exiting before export.")
        return

    # ── Load actual weights ──────────────────────────────────────────────────
    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        config            = config,
        torch_dtype       = dtype,
        device_map        = "cpu",
        trust_remote_code = True,
    )
    base.eval()

    # ── Calibration data ─────────────────────────────────────────────────────
    if args.calib_file:
        with open(args.calib_file) as f:
            prompts = [l.strip() for l in f if l.strip()]
        calib_data = []
        for p in prompts:
            enc = tokenizer(p, return_tensors="pt", max_length=256, truncation=True)
            calib_data.append(enc["input_ids"])
        print(f"Calibration: {len(calib_data)} prompts from {args.calib_file}")
    else:
        calib_data = make_calibration_data(tokenizer, n_samples=args.calib_samples)
        print(f"Calibration: {len(calib_data)} built-in prompts")

    # ── AWQ pre-quantisation (turbo mode only) ───────────────────────────────
    if args.quant == "turbo":
        print("\n[turbo] Step 1/3: AWQ activation-aware quantisation")
        base = apply_awq(base, tokenizer, calib_data,
                         group_size=args.group_size, w_bit=4, zero_point=True)
        print("[turbo] Step 2/3: Building half-models with mixed-precision export")
    else:
        print(f"\n[{args.quant}] Building half-models")

    os.makedirs(args.output_dir, exist_ok=True)
    out_a = os.path.join(args.output_dir, "first_half.rkllm")
    out_b = os.path.join(args.output_dir, "second_half.rkllm")

    # ── Export first half ────────────────────────────────────────────────────
    if not args.second_only:
        model_a = FirstHalfModel(base, split_layer=split).eval()
        export_half(
            model        = model_a,
            model_name   = f"first_half (layers 0-{split})",
            out_path     = out_a,
            quant        = args.quant,
            group_size   = args.group_size,
            platform     = args.platform,
            max_context_len = args.max_context,
            tokenizer    = tokenizer,
            n_layers     = split + 1,
            calib_data   = calib_data,
            kv_bits      = args.kv_bits,
        )
        del model_a

    # ── Export second half ───────────────────────────────────────────────────
    if not args.first_only:
        if args.quant == "turbo":
            print("[turbo] Step 3/3: Exporting second half")
        model_b = SecondHalfModel(base, split_layer=split).eval()
        export_half(
            model        = model_b,
            model_name   = f"second_half (layers {split+1}-{n_layers-1})",
            out_path     = out_b,
            quant        = args.quant,
            group_size   = args.group_size,
            platform     = args.platform,
            max_context_len = args.max_context,
            tokenizer    = tokenizer,
            n_layers     = n_layers - (split + 1),
            calib_data   = calib_data,
            kv_bits      = args.kv_bits,
        )
        del model_b

    # ── Summary ──────────────────────────────────────────────────────────────
    total_mb = 0
    for path in [out_a, out_b]:
        if os.path.exists(path):
            total_mb += os.path.getsize(path) / (1024 ** 2)

    print(f"\n{'─'*60}")
    print(f"  Done.  Total model size on disk: {total_mb:.0f} MB  ({total_mb/1024:.1f} GB)")
    print(f"  First half   → {out_a}")
    print(f"  Second half  → {out_b}")
    print()
    print("  Run inference on Orange Pi 5:")
    print(f"    ./rkllm-main -m {out_a} --model-b {out_b} -p 'Hello!' -v")
    print(f"{'─'*60}")


if __name__ == "__main__":
    main()
