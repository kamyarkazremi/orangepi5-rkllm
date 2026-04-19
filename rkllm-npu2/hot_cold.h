#pragma once
/*
 * rkllm-npu2/hot_cold.h
 *
 * Option B – True PowerInfer-style hot/cold neuron routing on RK3588
 * using the low-level RKNPU2 SDK (rknn_api.h) instead of RKLLM.
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * Overview
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * PowerInfer (GPU version) splits each FFN layer into:
 *   hot neurons  (~30% activated most often)  → GPU VRAM
 *   cold neurons (~70% rarely activated)       → CPU RAM
 *   predictor network                          → GPU (fast lookup)
 *
 * This file describes how to map the same idea onto RK3588:
 *
 *   hot neurons   → RK3588 NPU  (via RKNPU2 / rknn_api.h)
 *   cold neurons  → RK3588 CPU  (Cortex-A76 big cores via GGML/NEON)
 *   predictor     → NPU or CPU  (small MLP, fast either way)
 *   attention      → NPU        (entire attention block per layer)
 *   sampling       → CPU
 *
 * The NPU and CPU can run CONCURRENTLY on RK3588 — they are independent
 * hardware blocks.  So cold-neuron CPU compute and next-layer NPU attention
 * can overlap.
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * Per-layer execution flow
 * ═══════════════════════════════════════════════════════════════════════════
 *
 *  ┌────────────────────────────────────────────────────────┐
 *  │  Layer i  (repeat for each transformer layer)          │
 *  │                                                        │
 *  │  [CPU]  x = residual_stream[i]                        │
 *  │                                                        │
 *  │  ── Attention block ────────────────────────────────── │
 *  │  [NPU]  attn_out = attention(x, kv_cache[i])          │
 *  │  [CPU]  wait for attn_out                             │
 *  │  [CPU]  h = x + attn_out          (residual add)      │
 *  │                                                        │
 *  │  ── FFN hot/cold split ─────────────────────────────── │
 *  │  [CPU/NPU] mask = predictor(h)    (hot-neuron mask)   │
 *  │                                                        │
 *  │  [NPU async]  hot_out  = ffn_hot(h,  mask)  ──────┐   │
 *  │  [CPU async]  cold_out = ffn_cold(h, ~mask) ──┐   │   │
 *  │                                                │   │   │
 *  │  (NPU and CPU run in parallel here)            │   │   │
 *  │                                                │   │   │
 *  │  [CPU]  wait for both      ◄───────────────────┘   │   │
 *  │                            ◄───────────────────────┘   │
 *  │  [CPU]  ffn_out = hot_out + cold_out  (merge)          │
 *  │  [CPU]  residual_stream[i+1] = h + ffn_out            │
 *  └────────────────────────────────────────────────────────┘
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * Model preparation (offline, on x86 host)
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * For each transformer layer l:
 *
 *   1. Compute neuron activation frequencies over a calibration corpus.
 *      hot_mask[l][j] = 1  iff  neuron j fires in top-K% of tokens
 *
 *   2. Export per-layer RKNN models:
 *      attn_l.rknn        – attention block for layer l
 *      ffn_hot_l.rknn     – W_up[hot_mask],  W_gate[hot_mask], W_down[hot_mask]
 *      predictor_l.rknn   – small 2-layer MLP that predicts hot_mask at runtime
 *
 *   3. Export per-layer cold weights for GGML:
 *      ffn_cold_l.ggml    – W_up[~hot_mask], W_gate[~hot_mask], W_down[~hot_mask]
 *      (loaded into CPU RAM as fp16 / q4 tensors)
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * C interface (to be implemented in rkllm-npu2/hot_cold.c)
 * ═══════════════════════════════════════════════════════════════════════════
 */

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// Opaque context types
// ---------------------------------------------------------------------------

typedef struct hc_layer_ctx  hc_layer_ctx;   // per-transformer-layer context
typedef struct hc_model_ctx  hc_model_ctx;   // whole-model context

// ---------------------------------------------------------------------------
// Layer configuration
// ---------------------------------------------------------------------------

typedef struct {
    const char * attn_rknn_path;      // path to attention RKNN model
    const char * ffn_hot_rknn_path;   // path to hot-FFN RKNN model
    const char * pred_rknn_path;      // path to predictor RKNN model
    const char * ffn_cold_ggml_path;  // path to cold-FFN GGML weight file

    int32_t  hidden_dim;              // model hidden dimension (e.g. 4096)
    int32_t  ffn_dim;                 // FFN intermediate dimension (e.g. 11008)
    int32_t  n_hot;                   // number of hot neurons in this layer
    int32_t  n_cold;                  // number of cold neurons in this layer
    int32_t  n_heads;                 // number of attention heads
    int32_t  head_dim;                // dimension per head
    int32_t  max_seq_len;             // max sequence length (for KV-cache)
} hc_layer_cfg;

// ---------------------------------------------------------------------------
// Model-level init / teardown
// ---------------------------------------------------------------------------

// Initialise the hot/cold model.  Loads all RKNN models and GGML weights.
// n_layers:  number of transformer layers
// cfgs:      array of n_layers configurations, one per layer
// n_threads: number of CPU threads for cold-neuron GGML compute
hc_model_ctx * hc_model_init(int32_t           n_layers,
                              const hc_layer_cfg * cfgs,
                              int32_t           n_threads);

void hc_model_free(hc_model_ctx * ctx);

// ---------------------------------------------------------------------------
// KV-cache management
// ---------------------------------------------------------------------------

// Clear the NPU-side KV-cache for all attention layers.
void hc_model_clear_kv_cache(hc_model_ctx * ctx);

// ---------------------------------------------------------------------------
// Inference
// ---------------------------------------------------------------------------

// Run one full forward pass for n_tokens new tokens.
//
// token_ids:     int32_t[n_tokens]    – new token IDs (prefill or single decode)
// logits_out:    float[vocab_size]    – written on return; caller-allocated
// vocab_size:    vocabulary size
// keep_history:  1 = extend KV-cache from previous call (decode step)
//                0 = fresh context (first call / after clear)
//
// Returns 0 on success.
int hc_model_run(hc_model_ctx * ctx,
                 const int32_t * token_ids,
                 int32_t         n_tokens,
                 float         * logits_out,
                 int32_t         vocab_size,
                 int32_t         keep_history);

// ---------------------------------------------------------------------------
// Per-layer hot/cold split (called internally by hc_model_run)
// ---------------------------------------------------------------------------

// Run one transformer layer using the hot/cold split:
//   1. NPU: attention
//   2. NPU async: hot FFN neurons
//   3. CPU async: cold FFN neurons   (concurrent with step 2)
//   4. CPU: merge hot + cold, residual add
//
// hidden_in:   float[n_tokens * hidden_dim]  – input hidden states
// hidden_out:  float[n_tokens * hidden_dim]  – output hidden states (caller alloc)
// layer_idx:   which layer to run
// keep_history: as above
int hc_layer_run(hc_model_ctx * ctx,
                 const float  * hidden_in,
                 float        * hidden_out,
                 int32_t        n_tokens,
                 int32_t        layer_idx,
                 int32_t        keep_history);

// ---------------------------------------------------------------------------
// Predictor
// ---------------------------------------------------------------------------

// Run the predictor MLP for layer l on the current hidden state.
// Writes a binary hot-neuron mask: mask[j] = 1 iff neuron j is hot.
// mask: uint8_t[ffn_dim] – caller-allocated
int hc_predict_hot_mask(hc_model_ctx * ctx,
                         const float  * hidden,   // float[hidden_dim]
                         uint8_t      * mask,
                         int32_t        layer_idx);

// ---------------------------------------------------------------------------
// Hardware notes for implementors
// ---------------------------------------------------------------------------
//
// RKNPU2 SDK (rknn_api.h) functions you will need:
//
//   rknn_init()           – load a .rknn model into the NPU
//   rknn_run()            – synchronous NPU forward pass
//   rknn_inputs_set()     – copy float[] input to NPU DMA buffer
//   rknn_outputs_get()    – copy float[] output from NPU DMA buffer
//   rknn_outputs_release()
//   rknn_destroy()
//
// GGML functions for cold-FFN on CPU:
//
//   ggml_mul_mat()        – W_up @ x,  W_gate @ x
//   ggml_silu()           – SiLU gate activation
//   ggml_mul()            – element-wise gate * up
//   ggml_mul_mat()        – W_down @ (gate * up)
//
// Parallelism:
//   Launch NPU (hot) job via rknn_run() — it returns before the NPU finishes.
//   Meanwhile run GGML cold compute on CPU worker threads.
//   After both complete, merge on CPU with ggml_add().
//
//   Note: rknn_run() is *synchronous* in the current SDK, so to overlap you
//   must either:
//     (a) use rknn_run() in a dedicated NPU thread and do CPU work on main, or
//     (b) split into rknn_inputs_set() + rknn_run() in a pthread, then join.
//   Option (a) is simpler and recommended.

#ifdef __cplusplus
}
#endif
