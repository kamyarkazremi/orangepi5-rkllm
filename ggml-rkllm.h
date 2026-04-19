#pragma once

// RKLLM CPU/NPU hybrid backend for PowerInfer
// Target: RK3588 / RK3588S (Orange Pi 5, Rock 5B, ...)
//
// Split strategy (using official RKLLM SDK features):
//
//   CPU side:
//     - Tokenisation (PowerInfer tokeniser)
//     - Optional: embedding lookup from GGUF weights on CPU RAM
//     - Custom sampling: temperature, top-k, top-p, repetition penalty
//     - KV-cache / context management decisions
//
//   NPU side (RKLLM runtime on RK3588 NPU):
//     - Embedding lookup   (when using RKLLM_INPUT_TOKEN mode)
//     - All transformer layers: attention + FFN
//     - Logit projection
//     - Returns raw logits to CPU via callback (RKLLM_INFER_GET_LOGITS)
//
// Two input modes are supported:
//   TOKEN mode  – pass token IDs; NPU does the embedding lookup too
//   EMBED mode  – CPU computes embedding vectors and passes float[] to NPU
//
// RKLLM SDK reference: https://github.com/airockchip/rknn-llm

#include "ggml.h"
#include "ggml-backend.h"

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// Re-declaration of RKLLM SDK types (subset used by this backend).
// When building with GGML_USE_RKLLM on the target device, the real rkllm.h
// is included by ggml-rkllm.c and these stubs are not compiled.
// ---------------------------------------------------------------------------

#ifndef RKLLM_H
typedef void * LLMHandle;

// --- Result structures returned through the callback ---

typedef struct {
    float   * logits;       // float[vocab_size]  — valid only during callback
    int32_t   num_tokens;   // number of tokens whose logits are returned
    int32_t   vocab_size;
} RKLLMResultLogits;

typedef struct {
    float   * hidden_states; // float[num_tokens * embd_size]
    int32_t   num_tokens;
    int32_t   embd_size;
} RKLLMResultLastHiddenLayer;

typedef struct {
    char    * text;                             // decoded text for this step (may be NULL)
    int32_t   token_id;                         // raw token id
    RKLLMResultLogits          * logits;        // non-NULL when mode == RKLLM_INFER_GET_LOGITS
    RKLLMResultLastHiddenLayer * last_hidden_layer; // non-NULL when mode == RKLLM_INFER_GET_LAST_HIDDEN_LAYER
} RKLLMResult;

typedef enum {
    LLM_RUN_NORMAL  = 0,
    LLM_RUN_WAITING = 1,
    LLM_RUN_FINISH  = 2,
    LLM_RUN_ERROR   = 3,
} LLMCallState;

typedef int32_t (*LLMResultCallback)(RKLLMResult * result, void * userdata, LLMCallState state);

// --- Input structures ---

typedef enum {
    RKLLM_INPUT_PROMPT     = 0,  // null-terminated string
    RKLLM_INPUT_TOKEN      = 1,  // int32_t[] token IDs
    RKLLM_INPUT_EMBED      = 2,  // float[]  pre-computed embeddings from CPU
    RKLLM_INPUT_MULTIMODAL = 3,
} RKLLMInputType;

typedef struct {
    int32_t * input_ids;
    size_t    n_tokens;
} RKLLMTokenInput;

typedef struct {
    float  * embed_input;    // float[n_tokens * embed_dim]
    size_t   n_tokens;
} RKLLMEmbedInput;

typedef struct {
    RKLLMInputType input_type;
    union {
        const char      * prompt_input;
        RKLLMTokenInput   token_input;
        RKLLMEmbedInput   embed_input;
    };
} RKLLMInput;

// --- Inference mode ---

typedef enum {
    RKLLM_INFER_GENERATE              = 0, // standard autoregressive generation
    RKLLM_INFER_GET_LOGITS            = 1, // return raw logits to CPU callback
    RKLLM_INFER_GET_LAST_HIDDEN_LAYER = 2, // return last hidden state to CPU
} LLMInferMode;

typedef struct {
    LLMInferMode mode;
    int32_t      keep_history;  // 1 = keep KV-cache from previous run
} RKLLMInferParam;

// --- Extended / init params ---

typedef struct {
    int32_t base_domain_id; // NPU domain; 0 = default
    uint8_t reserved[60];
} LLMExtendParam;

typedef struct {
    const char    * model_path;
    int32_t         max_context_len;
    int32_t         max_new_tokens;   // -1 = unlimited
    int32_t         top_k;
    float           top_p;
    float           temperature;
    float           repeat_penalty;
    float           frequency_penalty;
    float           presence_penalty;
    bool            skip_special_token;
    bool            is_async;
    int32_t         num_npu_core;     // 1-3 on RK3588
    const char    * img_start;        // multimodal only
    const char    * img_end;
    const char    * img_content;
    LLMExtendParam  extend_param;
} RKLLMParam;

#endif // RKLLM_H

// ---------------------------------------------------------------------------
// Context-window compression config  (StreamingLLM-style sliding window)
// ---------------------------------------------------------------------------
//
// Problem: RKLLM's KV-cache has a hard limit (max_context_len).  Once full,
// generation stops or wraps destructively.
//
// Solution (StreamingLLM / attention-sink technique, MIT + Together AI 2023):
//
//   The model always pays disproportionate attention to the first few tokens
//   ("attention sinks") even when they carry no information.  If we always
//   keep those sink tokens in the KV-cache, we can evict all other old tokens
//   and slide a rolling window of recent tokens — giving effectively infinite
//   context with a fixed-size KV-cache.
//
//   Memory usage: constant at  (sink_tokens + window_tokens) * kv_bytes_per_tok
//   Context quality: retains local coherence + global anchoring via sinks.
//
// Also: when split mode is active, the hidden states that pass between
// NPU-A and NPU-B are compressed from fp16 → INT8 on the CPU (TurboQuant-
// style vector quantisation).  This halves the inter-model buffer size and
// reduces peak CPU RAM usage by hidden_dim * n_tokens * 1 byte vs 2 bytes.
//
//   KV-cache memory reduction (RK3588, 7B model, 2048-token window):
//     Without compression : ~512 MB  (fp16 KV, managed by RKLLM internally)
//     With sliding window : constant regardless of total tokens generated
//     Hidden-state buffer : fp16 = 32 KB/tok  →  INT8 = 16 KB/tok  (split mode)

typedef enum {
    RKLLM_CTX_TRUNCATE = 0,   // default: hard stop at max_context_len
    RKLLM_CTX_SLIDING  = 1,   // StreamingLLM: sink tokens + rolling window
} RKLLMCtxMode;

typedef struct {
    RKLLMCtxMode mode;

    // --- sliding window parameters (only used when mode == RKLLM_CTX_SLIDING) ---
    int32_t sink_tokens;    // number of initial tokens always kept in KV-cache.
                            // default 4.  These are the "attention sinks" —
                            // tokens the model dumps residual attention weight
                            // onto.  Keeping them stabilises generation.
    int32_t window_tokens;  // rolling window of recent tokens kept alongside
                            // the sinks.  Total KV-cache slots used:
                            //   sink_tokens + window_tokens  ≤ max_context_len
                            // default = max_context_len - sink_tokens.

    // --- hidden-state compression (split mode only) ---
    bool    compress_hidden; // compress fp16 hidden states to INT8 before
                             // passing between NPU-A and NPU-B.
                             // 2× smaller inter-model buffer on CPU RAM.
                             // default true when split mode is active.
} RKLLMCtxCfg;

// Default config — no sliding window, INT8 compression on in split mode.
static inline RKLLMCtxCfg rkllm_ctx_cfg_default(void) {
    RKLLMCtxCfg c;
    c.mode            = RKLLM_CTX_TRUNCATE;
    c.sink_tokens     = 4;
    c.window_tokens   = 0;      // 0 = auto (max_context_len - sink_tokens)
    c.compress_hidden = true;
    return c;
}

// ---------------------------------------------------------------------------
// Hybrid context – one per loaded model pair
// ---------------------------------------------------------------------------

typedef struct ggml_rkllm_ctx ggml_rkllm_ctx;

// ---------------------------------------------------------------------------
// Init / teardown
// ---------------------------------------------------------------------------

// Create a hybrid context from the FIRST-HALF (or full) model.
// model_path:   path to a .rkllm file (layers 0..split, or the full model)
// num_npu_core: 1-3; use 3 for maximum throughput on RK3588
// callback:     optional – called for text tokens in GENERATE mode only;
//               logit/hidden-state capture is handled internally.
// Returns NULL on failure.
GGML_API ggml_rkllm_ctx * ggml_rkllm_init(
    const char      * model_path,
    int32_t           num_npu_core,
    int32_t           max_context_len,
    int32_t           max_new_tokens,
    float             temperature,
    float             top_p,
    int32_t           top_k,
    float             repeat_penalty,
    LLMResultCallback callback   // NULL = built-in stdout printer
);

// Load the SECOND-HALF model (layers split+1..end) to enable the layer-split.
//
// After this call the inference path becomes:
//   token_ids → handle_a (GET_LAST_HIDDEN_LAYER)
//             → hidden_states[] on CPU
//             → handle_b (RKLLM_INPUT_EMBED, GET_LOGITS)
//             → logits[] on CPU  → CPU sampler
//
// Returns 0 on success, -1 on failure.
GGML_API int  ggml_rkllm_load_second_half(
    ggml_rkllm_ctx * ctx,
    const char     * model_b_path,
    int32_t          num_npu_core
);

// Returns true if ctx is valid and handle_a is loaded.
GGML_API bool ggml_rkllm_is_loaded(const ggml_rkllm_ctx * ctx);

// Free all resources (both handles).
GGML_API void ggml_rkllm_free(ggml_rkllm_ctx * ctx);

// ---------------------------------------------------------------------------
// NPU inference – TOKEN input
// ---------------------------------------------------------------------------
//
// In single-model mode:  handle_a runs the full model, returns logits.
// In layer-split mode:   handle_a runs layers 0..split (GET_LAST_HIDDEN_LAYER),
//                        then handle_b runs layers split+1..end (GET_LOGITS).
//
// infer_mode:    only used in single-model mode; ignored in split mode.
// keep_history:  1 = extend the KV-cache from the previous call (decode step).
//                0 = start a fresh context (prefill / first call).
// Returns 0 on success.
GGML_API int ggml_rkllm_run_tokens(
    ggml_rkllm_ctx * ctx,
    const int32_t  * token_ids,
    int32_t          n_tokens,
    LLMInferMode     infer_mode,
    int32_t          keep_history,
    void           * userdata
);

// ---------------------------------------------------------------------------
// NPU inference – EMBED input (CPU pre-computed embeddings → second half)
// ---------------------------------------------------------------------------
//
// embed:     float[n_tokens * embed_dim] computed on CPU
// embed_dim: model embedding dimension (e.g. 4096 for a 7B model)
// In split mode this goes to handle_b; in single mode it goes to handle_a.
// Returns 0 on success.
GGML_API int ggml_rkllm_run_embed(
    ggml_rkllm_ctx * ctx,
    const float    * embed,
    int32_t          n_tokens,
    int32_t          embed_dim,
    LLMInferMode     infer_mode,
    int32_t          keep_history,
    void           * userdata
);

// ---------------------------------------------------------------------------
// NPU inference – raw text prompt (convenience, no CPU sampling)
// ---------------------------------------------------------------------------

GGML_API int ggml_rkllm_run_prompt(
    ggml_rkllm_ctx * ctx,
    const char     * prompt,
    LLMInferMode     infer_mode,
    int32_t          keep_history,
    void           * userdata
);

// ---------------------------------------------------------------------------
// CPU sampler
// ---------------------------------------------------------------------------

// Sample one token from the logits captured during the last NPU run.
// Applies: repetition penalty → temperature → softmax → top-k/top-p.
// The chosen token is automatically appended to the rep-penalty history.
// Returns the token id, or -1 if no logits are available.
GGML_API int32_t ggml_rkllm_sample(ggml_rkllm_ctx * ctx);

// Read-only access to the raw logits captured from the last NPU run.
// Pointer is valid until the next ggml_rkllm_run_* call.
GGML_API const float * ggml_rkllm_get_logits(
    const ggml_rkllm_ctx * ctx,
    int32_t              * out_vocab_size   // may be NULL
);

// Read-only access to the hidden states returned by handle_a.
// Pointer is valid until the next handle_a run.
GGML_API const float * ggml_rkllm_get_hidden(
    const ggml_rkllm_ctx * ctx,
    int32_t              * out_n_tokens,    // may be NULL
    int32_t              * out_embd_dim     // may be NULL
);

// ---------------------------------------------------------------------------
// Context-window compression
// ---------------------------------------------------------------------------

// Apply a context-compression configuration to ctx.
// Call after ggml_rkllm_init() and before the first ggml_rkllm_run_tokens().
// cfg.window_tokens == 0 → auto-set to (max_context_len - sink_tokens).
GGML_API void ggml_rkllm_set_ctx_cfg(ggml_rkllm_ctx    * ctx,
                                      const RKLLMCtxCfg * cfg);

// Return the current context config.
GGML_API RKLLMCtxCfg ggml_rkllm_get_ctx_cfg(const ggml_rkllm_ctx * ctx);

// How many tokens are currently in the NPU KV-cache.
GGML_API int32_t ggml_rkllm_ctx_tokens(const ggml_rkllm_ctx * ctx);

// ---------------------------------------------------------------------------
// Control
// ---------------------------------------------------------------------------

// Abort an in-flight generation (both handles).
GGML_API void ggml_rkllm_abort(ggml_rkllm_ctx * ctx);

// Clear KV-cache on both handles (start a completely fresh context).
GGML_API void ggml_rkllm_clear_kv_cache(ggml_rkllm_ctx * ctx);

// ---------------------------------------------------------------------------
// ggml_backend interface (advertises the backend to the GGML scheduler)
// ---------------------------------------------------------------------------

GGML_API ggml_backend_t ggml_backend_rkllm_init(const char * model_path);
GGML_API bool           ggml_backend_is_rkllm(ggml_backend_t backend);

#ifdef __cplusplus
}
#endif
