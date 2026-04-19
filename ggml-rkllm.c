// ggml-rkllm.c  –  CPU/NPU hybrid backend for PowerInfer on RK3588 (Orange Pi 5)
//
// Build:  cmake -DLLAMA_RKLLM=ON -DRKLLM_ROOT=/path/to/rknn-llm/rkllm-runtime
//
// Layer-split inference flow (two-model mode):
//
//   [CPU]  token_ids
//     |
//     v  RKLLM_INPUT_TOKEN + RKLLM_INFER_GET_LAST_HIDDEN_LAYER
//   [NPU]  handle_a  (layers 0 .. split_layer)
//     |
//     v  hidden_states[] copied to CPU RAM
//   [CPU]  (optional: additional CPU ops on hidden state)
//     |
//     v  RKLLM_INPUT_EMBED + RKLLM_INFER_GET_LOGITS
//   [NPU]  handle_b  (layers split_layer+1 .. end)
//     |
//     v  logits[] copied to CPU RAM
//   [CPU]  rep-penalty → temperature → softmax → top-k/top-p → token id
//     |
//     `--- feed back as next input (keep_history=1)
//
// Single-model mode (handle_b == NULL):
//   handle_a runs the full model with RKLLM_INFER_GET_LOGITS; CPU still samples.

#include "ggml-rkllm.h"
#include "ggml-backend-impl.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <pthread.h>

// Pull in the real SDK only when building on the target device.
#ifdef GGML_USE_RKLLM
#  include <rkllm.h>
#else
// Stub implementations so the file compiles on dev / CI machines.
static inline int  rkllm_init(LLMHandle *h, RKLLMParam *p, LLMResultCallback cb) {
    (void)h; (void)p; (void)cb; return -1;
}
static inline int  rkllm_run(LLMHandle h, RKLLMInput *in, RKLLMInferParam *ip, void *ud) {
    (void)h; (void)in; (void)ip; (void)ud; return -1;
}
static inline int  rkllm_abort(LLMHandle h) { (void)h; return -1; }
static inline int  rkllm_clear_kv_cache(LLMHandle h, uint8_t m) { (void)h; (void)m; return -1; }
static inline int  rkllm_destroy(LLMHandle h) { (void)h; return -1; }
static inline RKLLMParam rkllm_createDefaultParam(void) {
    RKLLMParam p; memset(&p, 0, sizeof(p)); return p;
}
#endif

// ---------------------------------------------------------------------------
// Internal context
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Sliding-window token ring buffer
// ---------------------------------------------------------------------------
// Stores the full token history so we can re-encode after a cache eviction.
// Layout: [ sink_tokens ... | window (ring) ]
//
// We keep a flat array of max capacity = sink_tokens + window_tokens.
// Entries beyond that are dropped (the window slides).

typedef struct {
    int32_t * buf;        // token id storage
    int32_t   cap;        // allocated capacity  = sink + window
    int32_t   sink;       // number of sink tokens kept at the front
    int32_t   win;        // rolling window capacity
    int32_t   n_sink;     // sink tokens currently filled (0..sink)
    int32_t   win_head;   // index of oldest window token (ring)
    int32_t   win_len;    // number of window tokens currently stored
    int32_t   total;      // total tokens ever pushed (for overflow detection)
} TokenRing;

static void ring_init(TokenRing * r, int32_t sink, int32_t win) {
    r->cap      = sink + win;
    r->buf      = (int32_t *)calloc((size_t)r->cap, sizeof(int32_t));
    r->sink     = sink;
    r->win      = win;
    r->n_sink   = 0;
    r->win_head = 0;
    r->win_len  = 0;
    r->total    = 0;
}

static void ring_free(TokenRing * r) { free(r->buf); r->buf = NULL; }

static void ring_push(TokenRing * r, int32_t token_id) {
    r->total++;
    if (r->n_sink < r->sink) {
        // Still filling sink region (first few tokens).
        r->buf[r->n_sink++] = token_id;
        return;
    }
    // Window region (circular).
    int32_t slot = (r->win_head + r->win_len) % r->win;
    r->buf[r->sink + slot] = token_id;
    if (r->win_len < r->win) {
        r->win_len++;
    } else {
        // Window full — advance head (oldest entry evicted).
        r->win_head = (r->win_head + 1) % r->win;
    }
}

// Copy the current window into a flat array.  Returns number of tokens.
static int32_t ring_snapshot(const TokenRing * r, int32_t * out, int32_t max_out) {
    int32_t n = 0;
    // Sinks first.
    for (int32_t i = 0; i < r->n_sink && n < max_out; i++)
        out[n++] = r->buf[i];
    // Window in order (oldest first).
    for (int32_t i = 0; i < r->win_len && n < max_out; i++) {
        int32_t slot = (r->win_head + i) % r->win;
        out[n++] = r->buf[r->sink + slot];
    }
    return n;
}

static void ring_clear(TokenRing * r) {
    r->n_sink   = 0;
    r->win_head = 0;
    r->win_len  = 0;
    r->total    = 0;
}

// ---------------------------------------------------------------------------
// INT8 hidden-state compression  (TurboQuant-style, CPU-side)
// ---------------------------------------------------------------------------
// Used in layer-split mode to compress the fp16 hidden states returned by
// handle_a before storing/forwarding them.  Halves the CPU RAM used by the
// inter-model buffer and reduces cache-line pressure on the A76 cores.
//
// Per-tensor symmetric quantisation: one scale per entire tensor.
// Scale = max(|x|) / 127.  Reconstruction error ≤ max(|x|) / 127.

static void f16_to_i8(const float * src, int8_t * dst, float * scale_out, int32_t n) {
    float mx = 0.0f;
    for (int32_t i = 0; i < n; i++) {
        float a = src[i] < 0.0f ? -src[i] : src[i];
        if (a > mx) mx = a;
    }
    float sc  = (mx > 1e-8f) ? mx / 127.0f : 1.0f;
    float inv = 1.0f / sc;
    for (int32_t i = 0; i < n; i++)
        dst[i] = (int8_t)(src[i] * inv + (src[i] >= 0.0f ? 0.5f : -0.5f));
    *scale_out = sc;
}

static void i8_to_f16(const int8_t * src, float * dst, float scale, int32_t n) {
    for (int32_t i = 0; i < n; i++) dst[i] = src[i] * scale;
}

// ---------------------------------------------------------------------------
// Internal context
// ---------------------------------------------------------------------------

struct ggml_rkllm_ctx {
    // --- NPU handles ---
    LLMHandle  handle_a;   // first-half model (or full model when handle_b==NULL)
    LLMHandle  handle_b;   // second-half model; NULL = single-model mode

    // --- callback synchronisation ---
    pthread_mutex_t mtx;
    pthread_cond_t  cv;
    bool            done;       // set true by callback on FINISH/ERROR

    // --- results written by the callback, consumed by the caller thread ---
    float   * logits;           // float[vocab_size], realloc'd as needed
    int32_t   vocab_size;

    float   * hidden;           // float[n_tok * embd_dim], realloc'd as needed
    int8_t  * hidden_i8;        // INT8 compressed version, realloc'd as needed
    float     hidden_scale;     // dequant scale for hidden_i8
    int32_t   hidden_n_tok;
    int32_t   hidden_embd_dim;

    LLMCallState cb_state;

    // --- optional user-facing text-streaming callback (GENERATE mode) ---
    LLMResultCallback user_cb;
    void            * user_data;

    // --- sampler settings ---
    float    temperature;
    float    top_p;
    int32_t  top_k;
    float    repeat_penalty;

    // repetition-penalty token history (ring buffer)
    int32_t * rep_hist;
    int32_t   rep_hist_len;
    int32_t   rep_hist_cap;

    // --- context compression ---
    RKLLMCtxCfg  ctx_cfg;
    int32_t      max_context_len;   // from init params
    TokenRing    token_ring;        // full token history for re-encoding
    int32_t      n_in_cache;        // tokens currently in the NPU KV-cache
    bool         ring_active;       // true once ring is initialised
};

// ---------------------------------------------------------------------------
// Unified callback – called by RKLLM on the NPU driver thread for both handles
// ---------------------------------------------------------------------------

static int32_t internal_cb(RKLLMResult * result, void * userdata, LLMCallState state) {
    ggml_rkllm_ctx * ctx = (ggml_rkllm_ctx *)userdata;

    pthread_mutex_lock(&ctx->mtx);
    ctx->cb_state = state;

    if (result) {
        // Capture logits (from handle_b or single-model handle_a).
        if (result->logits && result->logits->logits) {
            int32_t vs = result->logits->vocab_size;
            if (vs != ctx->vocab_size || !ctx->logits) {
                free(ctx->logits);
                ctx->logits     = (float *)malloc((size_t)vs * sizeof(float));
                ctx->vocab_size = vs;
            }
            memcpy(ctx->logits, result->logits->logits, (size_t)vs * sizeof(float));
        }

        // Capture hidden states (from handle_a in split mode).
        // CRITICAL: data pointer is only valid during this callback – copy immediately.
        if (result->last_hidden_layer && result->last_hidden_layer->hidden_states) {
            RKLLMResultLastHiddenLayer * hl = result->last_hidden_layer;
            size_t bytes = (size_t)hl->num_tokens * (size_t)hl->embd_size * sizeof(float);
            if (hl->num_tokens != ctx->hidden_n_tok ||
                hl->embd_size  != ctx->hidden_embd_dim || !ctx->hidden) {
                free(ctx->hidden);
                ctx->hidden          = (float *)malloc(bytes);
                ctx->hidden_n_tok    = hl->num_tokens;
                ctx->hidden_embd_dim = hl->embd_size;
            }
            memcpy(ctx->hidden, hl->hidden_states, bytes);
        }

        // Forward text tokens to user callback (text-streaming / GENERATE mode).
        if (ctx->user_cb) {
            ctx->user_cb(result, ctx->user_data, state);
        }
    }

    if (state == LLM_RUN_FINISH || state == LLM_RUN_ERROR) {
        ctx->done = true;
        pthread_cond_signal(&ctx->cv);
    }

    pthread_mutex_unlock(&ctx->mtx);
    return 0;
}

// Block the calling thread until the current rkllm_run() completes.
static LLMCallState wait_done(ggml_rkllm_ctx * ctx) {
    pthread_mutex_lock(&ctx->mtx);
    while (!ctx->done) pthread_cond_wait(&ctx->cv, &ctx->mtx);
    ctx->done = false;
    LLMCallState s = ctx->cb_state;
    pthread_mutex_unlock(&ctx->mtx);
    return s;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static RKLLMParam build_param(const char * path, int32_t cores,
                               int32_t ctx_len, int32_t max_tok,
                               float temp, float top_p, int32_t top_k,
                               float rep) {
    RKLLMParam p    = rkllm_createDefaultParam();
    p.model_path    = path;
    p.num_npu_core  = (cores  > 0) ? cores  : 3;
    p.max_context_len = (ctx_len > 0) ? ctx_len : 2048;
    p.max_new_tokens  = max_tok;
    p.temperature   = temp;
    p.top_p         = top_p;
    p.top_k         = top_k;
    p.repeat_penalty = rep;
    p.skip_special_token = false; // we do our own EOS detection
    p.is_async      = false;      // we manage the loop ourselves
    return p;
}

// Run one synchronous rkllm_run call and block until it finishes.
static int run_sync(ggml_rkllm_ctx * ctx, LLMHandle h,
                    RKLLMInput * in, LLMInferMode mode, int32_t keep_hist,
                    void * userdata) {
    RKLLMInferParam ip;
    memset(&ip, 0, sizeof(ip));
    ip.mode         = mode;
    ip.keep_history = keep_hist;

    ctx->user_data = userdata;
    ctx->done      = false;

    int ret = rkllm_run(h, in, &ip, ctx);
    if (ret != 0) return ret;

    return (wait_done(ctx) == LLM_RUN_ERROR) ? -1 : 0;
}

// ---------------------------------------------------------------------------
// Init / teardown
// ---------------------------------------------------------------------------

ggml_rkllm_ctx * ggml_rkllm_init(const char      * model_path,
                                   int32_t           num_npu_core,
                                   int32_t           max_context_len,
                                   int32_t           max_new_tokens,
                                   float             temperature,
                                   float             top_p,
                                   int32_t           top_k,
                                   float             repeat_penalty,
                                   LLMResultCallback callback) {
    if (!model_path || !model_path[0]) {
        fprintf(stderr, "[rkllm] model_path is empty\n");
        return NULL;
    }

    ggml_rkllm_ctx * ctx = (ggml_rkllm_ctx *)calloc(1, sizeof(*ctx));
    if (!ctx) return NULL;

    pthread_mutex_init(&ctx->mtx, NULL);
    pthread_cond_init(&ctx->cv, NULL);

    ctx->temperature      = (temperature    > 0.0f) ? temperature    : 0.8f;
    ctx->top_p            = (top_p          > 0.0f) ? top_p          : 0.95f;
    ctx->top_k            = (top_k          > 0)    ? top_k          : 40;
    ctx->repeat_penalty   = (repeat_penalty > 0.0f) ? repeat_penalty : 1.1f;
    ctx->user_cb          = callback;
    ctx->rep_hist_cap     = 512;
    ctx->rep_hist         = (int32_t *)malloc((size_t)ctx->rep_hist_cap * sizeof(int32_t));
    ctx->max_context_len  = (max_context_len > 0) ? max_context_len : 2048;
    ctx->ctx_cfg          = rkllm_ctx_cfg_default();
    ctx->n_in_cache       = 0;
    ctx->ring_active      = false;

    RKLLMParam p = build_param(model_path, num_npu_core, max_context_len,
                                max_new_tokens, ctx->temperature, ctx->top_p,
                                ctx->top_k, ctx->repeat_penalty);

    if (rkllm_init(&ctx->handle_a, &p, internal_cb) != 0) {
        fprintf(stderr, "[rkllm] failed to load model: %s\n", model_path);
        pthread_cond_destroy(&ctx->cv);
        pthread_mutex_destroy(&ctx->mtx);
        free(ctx->rep_hist);
        free(ctx);
        return NULL;
    }

    fprintf(stderr, "[rkllm] loaded '%s' on %d NPU core(s)\n",
            model_path, p.num_npu_core);
    return ctx;
}

int ggml_rkllm_load_second_half(ggml_rkllm_ctx * ctx,
                                  const char     * model_b_path,
                                  int32_t          num_npu_core) {
    if (!ctx || ctx->handle_b) return -1;

    RKLLMParam p = build_param(model_b_path, num_npu_core, 0, 0,
                                ctx->temperature, ctx->top_p,
                                ctx->top_k, ctx->repeat_penalty);

    if (rkllm_init(&ctx->handle_b, &p, internal_cb) != 0) {
        fprintf(stderr, "[rkllm] failed to load second-half model: %s\n", model_b_path);
        return -1;
    }

    fprintf(stderr, "[rkllm] layer-split mode: second half loaded from '%s'\n", model_b_path);
    return 0;
}

bool ggml_rkllm_is_loaded(const ggml_rkllm_ctx * ctx) {
    return ctx && ctx->handle_a;
}

void ggml_rkllm_free(ggml_rkllm_ctx * ctx) {
    if (!ctx) return;
    if (ctx->handle_b) rkllm_destroy(ctx->handle_b);
    if (ctx->handle_a) rkllm_destroy(ctx->handle_a);
    free(ctx->logits);
    free(ctx->hidden);
    free(ctx->hidden_i8);
    free(ctx->rep_hist);
    if (ctx->ring_active) ring_free(&ctx->token_ring);
    pthread_cond_destroy(&ctx->cv);
    pthread_mutex_destroy(&ctx->mtx);
    free(ctx);
}

// ---------------------------------------------------------------------------
// NPU inference – TOKEN input
// ---------------------------------------------------------------------------

int ggml_rkllm_run_tokens(ggml_rkllm_ctx * ctx,
                           const int32_t  * token_ids,
                           int32_t          n_tokens,
                           LLMInferMode     infer_mode,
                           int32_t          keep_history,
                           void           * userdata) {
    if (!ctx || !ctx->handle_a) return -1;

    // ── Sliding-window eviction check ────────────────────────────────────────
    if (ctx->ctx_cfg.mode == RKLLM_CTX_SLIDING && keep_history) {
        int32_t headroom = ctx->max_context_len - ctx->n_in_cache;
        if (headroom < n_tokens) {
            int ret = sliding_window_evict(ctx);
            if (ret != 0) return ret;
            // After eviction the KV-cache is already rebuilt with keep_history=0
            // so the actual run below must also use keep_history=1 to extend it.
        }
    }

    // ── Push tokens into ring buffer for future evictions ────────────────────
    if (ctx->ring_active) {
        for (int32_t i = 0; i < n_tokens; i++)
            ring_push(&ctx->token_ring, token_ids[i]);
    }

    RKLLMInput in;
    memset(&in, 0, sizeof(in));
    in.input_type            = RKLLM_INPUT_TOKEN;
    in.token_input.input_ids = (int32_t *)token_ids;
    in.token_input.n_tokens  = (size_t)n_tokens;

    if (!ctx->handle_b) {
        // Single-model: full model → logits to CPU.
        int ret = run_sync(ctx, ctx->handle_a, &in,
                           RKLLM_INFER_GET_LOGITS, keep_history, userdata);
        if (ret == 0) ctx->n_in_cache += n_tokens;
        return ret;
    }

    // ── Layer-split mode ─────────────────────────────────────────────────────
    // Step 1: first-half NPU run → hidden states to CPU.
    int ret = run_sync(ctx, ctx->handle_a, &in,
                       RKLLM_INFER_GET_LAST_HIDDEN_LAYER, keep_history, userdata);
    if (ret != 0) return ret;

    // Step 2: (Optional) INT8 compress hidden states on CPU before forwarding.
    //   fp16 → INT8 → fp16 round-trip.  Saves 50% of the inter-model buffer.
    const float * embed_ptr = ctx->hidden;
    int32_t       n_elem    = ctx->hidden_n_tok * ctx->hidden_embd_dim;

    if (ctx->ctx_cfg.compress_hidden && n_elem > 0) {
        // Realloc INT8 buffer if needed.
        if (!ctx->hidden_i8) {
            ctx->hidden_i8 = (int8_t *)malloc((size_t)n_elem);
        } else {
            // Check capacity (hidden dims can only grow on first prefill).
            ctx->hidden_i8 = (int8_t *)realloc(ctx->hidden_i8,
                                                 (size_t)n_elem);
        }
        f16_to_i8(ctx->hidden, ctx->hidden_i8, &ctx->hidden_scale, n_elem);
        // Decompress back into ctx->hidden so the embed call below works.
        i8_to_f16(ctx->hidden_i8, ctx->hidden, ctx->hidden_scale, n_elem);
    }

    // Step 3: pass (possibly INT8-compressed-then-restored) hidden states to
    //         second-half NPU as embedding input.
    ret = ggml_rkllm_run_embed(ctx,
                                ctx->hidden,
                                ctx->hidden_n_tok,
                                ctx->hidden_embd_dim,
                                RKLLM_INFER_GET_LOGITS,
                                keep_history,
                                userdata);
    if (ret == 0) ctx->n_in_cache += n_tokens;
    return ret;
}

// ---------------------------------------------------------------------------
// NPU inference – EMBED input
// ---------------------------------------------------------------------------

int ggml_rkllm_run_embed(ggml_rkllm_ctx * ctx,
                          const float    * embed,
                          int32_t          n_tokens,
                          int32_t          embed_dim,
                          LLMInferMode     infer_mode,
                          int32_t          keep_history,
                          void           * userdata) {
    if (!ctx || !ctx->handle_a) return -1;
    (void)embed_dim; // carried by the model; not needed in the input struct

    LLMHandle target = ctx->handle_b ? ctx->handle_b : ctx->handle_a;

    RKLLMInput in;
    memset(&in, 0, sizeof(in));
    in.input_type              = RKLLM_INPUT_EMBED;
    in.embed_input.embed_input = (float *)embed;
    in.embed_input.n_tokens    = (size_t)n_tokens;

    return run_sync(ctx, target, &in, infer_mode, keep_history, userdata);
}

// ---------------------------------------------------------------------------
// NPU inference – raw text prompt
// ---------------------------------------------------------------------------

int ggml_rkllm_run_prompt(ggml_rkllm_ctx * ctx,
                           const char     * prompt,
                           LLMInferMode     infer_mode,
                           int32_t          keep_history,
                           void           * userdata) {
    if (!ctx || !ctx->handle_a) return -1;

    RKLLMInput in;
    memset(&in, 0, sizeof(in));
    in.input_type   = RKLLM_INPUT_PROMPT;
    in.prompt_input = prompt;

    // Prompt mode only goes to handle_a; layer-split is TOKEN/EMBED only.
    return run_sync(ctx, ctx->handle_a, &in, infer_mode, keep_history, userdata);
}

// ---------------------------------------------------------------------------
// Control
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Context-compression public API
// ---------------------------------------------------------------------------

void ggml_rkllm_set_ctx_cfg(ggml_rkllm_ctx * ctx, const RKLLMCtxCfg * cfg) {
    if (!ctx || !cfg) return;
    ctx->ctx_cfg = *cfg;

    // Auto-size window if not set.
    if (ctx->ctx_cfg.window_tokens <= 0) {
        ctx->ctx_cfg.window_tokens =
            ctx->max_context_len - ctx->ctx_cfg.sink_tokens;
    }

    // (Re-)initialise the token ring.
    if (ctx->ring_active) ring_free(&ctx->token_ring);
    if (cfg->mode == RKLLM_CTX_SLIDING) {
        ring_init(&ctx->token_ring,
                  ctx->ctx_cfg.sink_tokens,
                  ctx->ctx_cfg.window_tokens);
        ctx->ring_active = true;
    } else {
        ctx->ring_active = false;
    }
}

RKLLMCtxCfg ggml_rkllm_get_ctx_cfg(const ggml_rkllm_ctx * ctx) {
    RKLLMCtxCfg def = rkllm_ctx_cfg_default();
    return ctx ? ctx->ctx_cfg : def;
}

int32_t ggml_rkllm_ctx_tokens(const ggml_rkllm_ctx * ctx) {
    return ctx ? ctx->n_in_cache : 0;
}

// ---------------------------------------------------------------------------
// Sliding-window cache eviction
// ---------------------------------------------------------------------------
// Called just before a run_tokens() call when the KV-cache is near full.
// 1. Snapshot the sink + recent-window tokens from the ring.
// 2. Clear both NPU KV-caches.
// 3. Re-encode those tokens as a fresh prefill (keep_history=0).
// After return, ctx->n_in_cache reflects the re-encoded token count.

static int sliding_window_evict(ggml_rkllm_ctx * ctx) {
    int32_t max_snap = ctx->ctx_cfg.sink_tokens + ctx->ctx_cfg.window_tokens;
    int32_t * snap   = (int32_t *)malloc((size_t)max_snap * sizeof(int32_t));
    int32_t   n_snap = ring_snapshot(&ctx->token_ring, snap, max_snap);

    if (n_snap == 0) { free(snap); return 0; }

    fprintf(stderr,
        "[ctx] window eviction: keeping %d sink + %d window tokens "
        "(evicted %d)\n",
        ctx->ctx_cfg.sink_tokens,
        n_snap - ctx->ctx_cfg.sink_tokens,
        ctx->n_in_cache - n_snap);

    // Clear both KV-caches.
    if (ctx->handle_a) rkllm_clear_kv_cache(ctx->handle_a, 0);
    if (ctx->handle_b) rkllm_clear_kv_cache(ctx->handle_b, 0);
    ctx->n_in_cache = 0;

    // Re-encode the retained tokens as a silent prefill (no user callback).
    RKLLMInput in;
    memset(&in, 0, sizeof(in));
    in.input_type            = RKLLM_INPUT_TOKEN;
    in.token_input.input_ids = snap;
    in.token_input.n_tokens  = (size_t)n_snap;

    // We just need the KV-cache populated; discard the logits.
    int ret = run_sync(ctx, ctx->handle_a, &in,
                       ctx->handle_b
                           ? RKLLM_INFER_GET_LAST_HIDDEN_LAYER
                           : RKLLM_INFER_GET_LOGITS,
                       /*keep_history=*/0, NULL);

    if (ret == 0 && ctx->handle_b) {
        // Pass hidden states to second half to rebuild its KV-cache too.
        RKLLMInput emb;
        memset(&emb, 0, sizeof(emb));
        emb.input_type              = RKLLM_INPUT_EMBED;
        emb.embed_input.embed_input = ctx->hidden;
        emb.embed_input.n_tokens    = (size_t)ctx->hidden_n_tok;
        ret = run_sync(ctx, ctx->handle_b, &emb,
                       RKLLM_INFER_GET_LOGITS, /*keep_history=*/0, NULL);
    }

    ctx->n_in_cache = n_snap;
    free(snap);
    return ret;
}

void ggml_rkllm_abort(ggml_rkllm_ctx * ctx) {
    if (!ctx) return;
    if (ctx->handle_a) rkllm_abort(ctx->handle_a);
    if (ctx->handle_b) rkllm_abort(ctx->handle_b);
}

void ggml_rkllm_clear_kv_cache(ggml_rkllm_ctx * ctx) {
    if (!ctx) return;
    if (ctx->handle_a) rkllm_clear_kv_cache(ctx->handle_a, 0);
    if (ctx->handle_b) rkllm_clear_kv_cache(ctx->handle_b, 0);
    ctx->n_in_cache = 0;
    if (ctx->ring_active) ring_clear(&ctx->token_ring);
}

// ---------------------------------------------------------------------------
// CPU sampler
// ---------------------------------------------------------------------------

static void push_rep_hist(ggml_rkllm_ctx * ctx, int32_t id) {
    if (ctx->rep_hist_len == ctx->rep_hist_cap) {
        memmove(ctx->rep_hist, ctx->rep_hist + 1,
                (size_t)(ctx->rep_hist_cap - 1) * sizeof(int32_t));
        ctx->rep_hist_len = ctx->rep_hist_cap - 1;
    }
    ctx->rep_hist[ctx->rep_hist_len++] = id;
}

static void apply_rep_penalty(float * logits, int32_t vocab_size,
                               const int32_t * hist, int32_t n, float penalty) {
    if (penalty == 1.0f || n == 0) return;
    for (int32_t i = 0; i < n; i++) {
        int32_t id = hist[i];
        if (id < 0 || id >= vocab_size) continue;
        logits[id] = (logits[id] > 0.0f) ? logits[id] / penalty
                                          : logits[id] * penalty;
    }
}

static void softmax_inplace(float * v, int32_t n) {
    float mx = -FLT_MAX;
    for (int32_t i = 0; i < n; i++) if (v[i] > mx) mx = v[i];
    float s = 0.0f;
    for (int32_t i = 0; i < n; i++) { v[i] = expf(v[i] - mx); s += v[i]; }
    float inv = 1.0f / s;
    for (int32_t i = 0; i < n; i++) v[i] *= inv;
}

typedef struct { float p; int32_t id; } Candidate;

static int cmp_desc(const void * a, const void * b) {
    float fa = ((Candidate *)a)->p, fb = ((Candidate *)b)->p;
    return (fb > fa) - (fb < fa);
}

static int32_t sample_top_kp(const float * probs, int32_t vs,
                               int32_t k, float p, uint64_t * rng) {
    Candidate * c = (Candidate *)malloc((size_t)vs * sizeof(Candidate));
    for (int32_t i = 0; i < vs; i++) { c[i].p = probs[i]; c[i].id = i; }
    qsort(c, (size_t)vs, sizeof(Candidate), cmp_desc);

    // top-k
    if (k > 0 && k < vs) vs = k;

    // top-p
    float cum = 0.0f;
    int32_t cut = vs;
    for (int32_t i = 0; i < vs; i++) {
        cum += c[i].p;
        if (cum >= p) { cut = i + 1; break; }
    }

    // renormalise
    float norm = 0.0f;
    for (int32_t i = 0; i < cut; i++) norm += c[i].p;

    // xorshift64 RNG
    *rng ^= *rng << 13; *rng ^= *rng >> 7; *rng ^= *rng << 17;
    float r = ((float)(*rng >> 11) / (float)(1ULL << 53)) * norm;

    int32_t chosen = c[cut - 1].id;
    cum = 0.0f;
    for (int32_t i = 0; i < cut; i++) {
        cum += c[i].p;
        if (r <= cum) { chosen = c[i].id; break; }
    }

    free(c);
    return chosen;
}

int32_t ggml_rkllm_sample(ggml_rkllm_ctx * ctx) {
    if (!ctx || !ctx->logits || ctx->vocab_size <= 0) return -1;

    float * logits = (float *)malloc((size_t)ctx->vocab_size * sizeof(float));
    memcpy(logits, ctx->logits, (size_t)ctx->vocab_size * sizeof(float));

    apply_rep_penalty(logits, ctx->vocab_size,
                      ctx->rep_hist, ctx->rep_hist_len, ctx->repeat_penalty);

    if (ctx->temperature > 0.0f && ctx->temperature != 1.0f) {
        float inv_t = 1.0f / ctx->temperature;
        for (int32_t i = 0; i < ctx->vocab_size; i++) logits[i] *= inv_t;
    }

    softmax_inplace(logits, ctx->vocab_size);

    static uint64_t rng = 0;
    if (!rng) rng = (uint64_t)time(NULL) ^ 0xdeadbeefcafeULL;

    int32_t tok = sample_top_kp(logits, ctx->vocab_size, ctx->top_k, ctx->top_p, &rng);
    free(logits);

    push_rep_hist(ctx, tok);
    return tok;
}

const float * ggml_rkllm_get_logits(const ggml_rkllm_ctx * ctx, int32_t * out_vocab_size) {
    if (!ctx) return NULL;
    if (out_vocab_size) *out_vocab_size = ctx->vocab_size;
    return ctx->logits;
}

const float * ggml_rkllm_get_hidden(const ggml_rkllm_ctx * ctx,
                                     int32_t * out_n_tokens,
                                     int32_t * out_embd_dim) {
    if (!ctx) return NULL;
    if (out_n_tokens) *out_n_tokens = ctx->hidden_n_tok;
    if (out_embd_dim) *out_embd_dim = ctx->hidden_embd_dim;
    return ctx->hidden;
}

// ---------------------------------------------------------------------------
// ggml_backend interface
// ---------------------------------------------------------------------------
// RKLLM runs at full-model granularity, not per-GGML-op.  This backend
// advertises itself so the GGML scheduler knows it exists, but returns false
// from supports_op() so all graph ops fall through to the CPU backend.
// Actual inference is driven by ggml_rkllm_run_tokens() called directly from
// examples/rkllm/main.cpp.

static const char * be_name(ggml_backend_t b) { (void)b; return "RKLLM-NPU"; }

static void be_free(ggml_backend_t b) {
    ggml_rkllm_ctx * ctx = (ggml_rkllm_ctx *)b->context;
    ggml_rkllm_free(ctx);
    free(b);
}

static ggml_backend_buffer_t be_alloc_buf(ggml_backend_t b, size_t sz) {
    (void)b;
    return ggml_backend_cpu_buffer_type()->alloc_buffer(ggml_backend_cpu_buffer_type(), sz);
}

static size_t be_alignment(ggml_backend_t b) { (void)b; return 64; }

static void be_set_tensor(ggml_backend_t b, struct ggml_tensor * t,
                           const void * d, size_t off, size_t sz) {
    (void)b; memcpy((char *)t->data + off, d, sz);
}
static void be_get_tensor(ggml_backend_t b, const struct ggml_tensor * t,
                           void * d, size_t off, size_t sz) {
    (void)b; memcpy(d, (const char *)t->data + off, sz);
}
static void be_sync(ggml_backend_t b) { (void)b; }
static void be_graph_compute(ggml_backend_t b, struct ggml_cgraph * g) { (void)b; (void)g; }
static bool be_supports_op(ggml_backend_t b, const struct ggml_tensor * op) {
    (void)b; (void)op; return false;
}

static struct ggml_backend_i rkllm_be_iface = {
    /* .get_name          = */ be_name,
    /* .free              = */ be_free,
    /* .alloc_buffer      = */ be_alloc_buf,
    /* .get_alignment     = */ be_alignment,
    /* .set_tensor_async  = */ be_set_tensor,
    /* .get_tensor_async  = */ be_get_tensor,
    /* .synchronize       = */ be_sync,
    /* .cpy_tensor_from   = */ NULL,
    /* .cpy_tensor_to     = */ NULL,
    /* .graph_plan_create = */ NULL,
    /* .graph_plan_free   = */ NULL,
    /* .graph_plan_compute= */ NULL,
    /* .graph_compute     = */ be_graph_compute,
    /* .supports_op       = */ be_supports_op,
};

ggml_backend_t ggml_backend_rkllm_init(const char * model_path) {
    ggml_rkllm_ctx * ctx = ggml_rkllm_init(model_path, 3, 2048, -1,
                                             0.8f, 0.95f, 40, 1.1f, NULL);
    if (!ctx) return NULL;

    ggml_backend_t b = (ggml_backend_t)calloc(1, sizeof(struct ggml_backend));
    if (!b) { ggml_rkllm_free(ctx); return NULL; }
    b->iface   = rkllm_be_iface;
    b->context = ctx;
    return b;
}

bool ggml_backend_is_rkllm(ggml_backend_t b) {
    return b && b->iface.get_name == be_name;
}
