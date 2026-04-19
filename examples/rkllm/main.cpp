// PowerInfer – RKLLM CPU/NPU hybrid inference entry point
// Target: Orange Pi 5 / Rock 5B (RK3588 / RK3588S NPU)
//
// ── Modes ───────────────────────────────────────────────────────────────────
//
//  Single-model  (-m model.rkllm)
//    NPU runs the full model.  CPU does sampling via RKLLM_INFER_GET_LOGITS.
//
//  Layer-split   (-m first_half.rkllm --model-b second_half.rkllm)
//    NPU-A  : layers  0..split  →  hidden states back to CPU
//    NPU-B  : layers split+1..end  (receives CPU hidden states as embeddings)
//             →  logits back to CPU
//    CPU    : rep-penalty → temperature → top-k/top-p sampling
//
// ── Usage ───────────────────────────────────────────────────────────────────
//   ./rkllm-main -m model.rkllm [--model-b second_half.rkllm] -p "Hello!"
//
// ── Model prep ──────────────────────────────────────────────────────────────
//   # On an x86 host with the RKLLM toolkit:
//   python scripts/split_rkllm.py --model hf/llama-3-8b \
//          --split-layer 16 --output-dir models/
//   # Produces models/first_half.rkllm and models/second_half.rkllm
//
// ── Build ────────────────────────────────────────────────────────────────────
//   cmake -S . -B build -DLLAMA_RKLLM=ON \
//         -DRKLLM_ROOT=/path/to/rknn-llm/rkllm-runtime
//   cmake --build build --target rkllm-main

#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <csignal>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <atomic>

#include "ggml-rkllm.h"

#ifdef GGML_USE_RKLLM
#include <rkllm.h>
#endif

// ---------------------------------------------------------------------------
// CLI parameters (subset of gpt_params relevant to RKLLM)
// ---------------------------------------------------------------------------

struct rkllm_params {
    std::string model;                   // first-half (or full) .rkllm model
    std::string model_b;                 // second-half model (layer-split mode)
    std::string prompt;
    std::string system_prompt;
    std::string input_prefix  = "### Human: ";
    std::string input_suffix  = "\n### Assistant:";
    int32_t     n_ctx         = 2048;
    int32_t     n_predict     = 512;
    int32_t     num_npu_core  = 3;
    float       temp          = 0.8f;
    float       top_p         = 0.9f;
    int32_t     top_k         = 40;
    float       repeat_penalty = 1.1f;
    int32_t     eos_token_id  = 2;       // model-specific; 2 = llama </s>
    bool        interactive   = false;
    bool        verbose       = false;

    // Context-window compression (StreamingLLM sliding window)
    bool        stream_ctx    = false;   // enable sliding-window context
    int32_t     sink_tokens   = 4;       // attention sinks to always keep
    int32_t     window_tokens = 0;       // rolling window size (0 = auto)
    bool        no_compress   = false;   // disable INT8 hidden-state compression
};

// ---------------------------------------------------------------------------
// Signal handling for graceful abort
// ---------------------------------------------------------------------------

static std::atomic<bool> g_abort_requested{false};
static ggml_rkllm_ctx  * g_ctx = nullptr;   // for signal handler

static void signal_handler(int signal) {
    (void)signal;
    g_abort_requested.store(true);
    if (g_ctx) ggml_rkllm_abort(g_ctx);
    fputc('\n', stdout);
}

// ---------------------------------------------------------------------------
// Timing helper
// ---------------------------------------------------------------------------

static double time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}

// ---------------------------------------------------------------------------
// Simple pseudo-tokeniser for prompt text
//
// In a full integration we would use llama.cpp's tokeniser (loading the GGUF
// model just for its vocabulary).  Here we use a byte-level fallback so the
// binary has no additional dependencies.  For real use, replace this with:
//
//   #include "llama.h"
//   llama_tokenize(model, text, tokens, max_tokens, /*add_bos=*/true);
// ---------------------------------------------------------------------------

// Encode text as UTF-8 bytes with an optional BOS token prepended.
// Returns a vector of token IDs.
static std::vector<int32_t> tokenize_simple(const std::string & text, int32_t bos_id = 1) {
    std::vector<int32_t> ids;
    if (bos_id >= 0) ids.push_back(bos_id);
    for (unsigned char c : text) ids.push_back(static_cast<int32_t>(c));
    return ids;
}

// ---------------------------------------------------------------------------
// CPU-sampled autoregressive decode loop
// ---------------------------------------------------------------------------
// Uses RKLLM_INFER_GET_LOGITS (single-model) or the layer-split path so that
// every sampling decision is made on the CPU ARM cores.

struct decode_stats {
    int32_t n_prefill  = 0;
    int32_t n_decode   = 0;
    double  t_prefill  = 0.0;  // ms
    double  t_first    = 0.0;  // ms to first token
    double  t_decode   = 0.0;  // ms for all decode steps
};

// Run one full prompt → generation cycle.
// Writes generated text to stdout.  Returns stats.
static decode_stats run_decode(ggml_rkllm_ctx       * ctx,
                                const rkllm_params   & p,
                                const std::string    & prompt_text) {
    decode_stats stats;

    // 1. Tokenise prompt on CPU.
    std::vector<int32_t> prompt_tokens = tokenize_simple(prompt_text, /*bos=*/1);
    stats.n_prefill = (int32_t)prompt_tokens.size();

    if (p.verbose) {
        fprintf(stderr, "[prefill] %d prompt tokens\n", stats.n_prefill);
    }

    // 2. Prefill: send prompt to NPU, get logits for the first output token.
    double t0 = time_ms();
    int ret = ggml_rkllm_run_tokens(ctx,
                                     prompt_tokens.data(),
                                     (int32_t)prompt_tokens.size(),
                                     RKLLM_INFER_GET_LOGITS,
                                     /*keep_history=*/0,
                                     /*userdata=*/nullptr);
    stats.t_prefill = time_ms() - t0;

    if (ret != 0) {
        fprintf(stderr, "[rkllm] prefill failed\n");
        return stats;
    }

    // 3. Sample first token on CPU.
    int32_t next_token = ggml_rkllm_sample(ctx);
    if (next_token < 0) {
        fprintf(stderr, "[rkllm] no logits returned from prefill\n");
        return stats;
    }
    stats.t_first = time_ms() - t0;

    // 4. Decode loop: feed one token at a time, keep KV-cache.
    double t_decode_start = time_ms();
    for (int32_t step = 0; step < p.n_predict; step++) {
        if (g_abort_requested.load()) break;
        if (next_token == p.eos_token_id)  break;

        // Decode token id → UTF-8 text.
        // With a real tokeniser: llama_token_to_piece(model, next_token)
        // Fallback: treat it as a raw byte character.
        if (next_token >= 32 && next_token < 127) {
            fputc((char)next_token, stdout);
        } else if (next_token > 127) {
            // Non-ASCII byte – print as hex escape so output stays valid.
            fprintf(stdout, "\\x%02x", (unsigned)next_token);
        }
        fflush(stdout);
        stats.n_decode++;

        // NPU forward pass for this single new token (KV-cache extended).
        ret = ggml_rkllm_run_tokens(ctx,
                                     &next_token, 1,
                                     RKLLM_INFER_GET_LOGITS,
                                     /*keep_history=*/1,
                                     nullptr);
        if (ret != 0) {
            fprintf(stderr, "\n[rkllm] decode step %d failed\n", step);
            break;
        }

        next_token = ggml_rkllm_sample(ctx);
        if (next_token < 0) break;
    }

    stats.t_decode = time_ms() - t_decode_start;
    fputc('\n', stdout);
    return stats;
}

static void print_stats(const decode_stats & s, bool verbose) {
    if (!verbose) return;
    double tok_per_s = (s.t_decode > 0.0 && s.n_decode > 0)
                       ? 1000.0 * s.n_decode / s.t_decode : 0.0;
    fprintf(stderr,
        "\n[timing]  prefill %d tok in %.1f ms  |"
        "  first-token %.1f ms  |"
        "  decode %d tok in %.1f ms  (%.1f tok/s)\n",
        s.n_prefill, s.t_prefill,
        s.t_first,
        s.n_decode,  s.t_decode,
        tok_per_s);
}

// ---------------------------------------------------------------------------
// CLI parsing
// ---------------------------------------------------------------------------

static void print_usage(const char * prog) {
    fprintf(stdout,
        "PowerInfer RKLLM CPU/NPU hybrid inference – Orange Pi 5 / RK3588\n\n"
        "Usage: %s -m model.rkllm [--model-b second_half.rkllm] [options]\n\n"
        "Options:\n"
        "  -m,  --model PATH        First-half (or full) .rkllm model (required)\n"
        "       --model-b PATH      Second-half .rkllm model (enables layer-split mode)\n"
        "  -p,  --prompt TEXT       Prompt text\n"
        "  -f,  --file PATH         Read prompt from file\n"
        "       --system TEXT       System / instruction prefix\n"
        "  -n,  --n-predict N       Max tokens to generate (default 512)\n"
        "  -c,  --ctx-size N        KV-cache length (default 2048)\n"
        "       --npu-cores N       NPU cores per model, 1-3 (default 3)\n"
        "       --temp N            Temperature (default 0.8)\n"
        "       --top-p N           Top-P (default 0.9)\n"
        "       --top-k N           Top-K (default 40)\n"
        "       --repeat-penalty N  Repeat penalty (default 1.1)\n"
        "       --eos-token N       EOS token id (default 2 = llama </s>)\n"
        "  -i,  --interactive       Chat / interactive mode\n"
        "       --input-prefix TEXT Prefix added before each user turn\n"
        "       --input-suffix TEXT Suffix added after each user turn\n"
        "  -v,  --verbose           Print timing info\n"
        "  -h,  --help              Show this help\n"
        "\n"
        "Context-window compression (StreamingLLM sliding window):\n"
        "       --stream-ctx        Enable sliding-window context (unlimited context\n"
        "                           with bounded KV-cache memory)\n"
        "       --sink-tokens N     Attention-sink token count (default 4).\n"
        "                           These initial tokens are always kept in the\n"
        "                           KV-cache to stabilise attention.\n"
        "       --window-tokens N   Rolling window size (default ctx-size - sinks).\n"
        "       --no-compress       Disable INT8 hidden-state compression between\n"
        "                           NPU-A and NPU-B (split mode only).\n"
        "\n"
        "Memory impact of --stream-ctx:\n"
        "  Without: KV-cache grows until max_context_len, then hard stop.\n"
        "  With:    KV-cache stays at (sink + window) tokens, generation\n"
        "           continues indefinitely with sliding context.\n"
        "\n"
        "Layer-split mode:\n"
        "  Export the model in two halves with scripts/split_rkllm.py, then:\n"
        "    ./rkllm-main -m models/first_half.rkllm --model-b models/second_half.rkllm\n"
        "  NPU-A runs layers 0..split, CPU receives hidden states, NPU-B finishes.\n"
        "  CPU performs sampling on the returned logits.\n"
        "\n",
        prog
    );
}

static bool parse_args(int argc, char ** argv, rkllm_params & p) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        auto next = [&]() -> const char * {
            if (i + 1 >= argc) {
                fprintf(stderr, "error: %s requires an argument\n", arg.c_str());
                exit(1);
            }
            return argv[++i];
        };

        if (arg == "-m" || arg == "--model")           p.model         = next();
        else if (arg == "--model-b")                   p.model_b       = next();
        else if (arg == "-p" || arg == "--prompt")     p.prompt        = next();
        else if (arg == "--system")                    p.system_prompt = next();
        else if (arg == "--input-prefix")              p.input_prefix  = next();
        else if (arg == "--input-suffix")              p.input_suffix  = next();
        else if (arg == "-f" || arg == "--file") {
            std::ifstream f(next());
            if (!f) { fprintf(stderr, "error: cannot open prompt file\n"); return false; }
            std::ostringstream ss;
            ss << f.rdbuf();
            p.prompt = ss.str();
        }
        else if (arg == "-n" || arg == "--n-predict")  p.n_predict     = atoi(next());
        else if (arg == "-c" || arg == "--ctx-size")   p.n_ctx         = atoi(next());
        else if (arg == "--npu-cores")                 p.num_npu_core  = atoi(next());
        else if (arg == "--temp")                      p.temp          = atof(next());
        else if (arg == "--top-p")                     p.top_p         = atof(next());
        else if (arg == "--top-k")                     p.top_k         = atoi(next());
        else if (arg == "--repeat-penalty")            p.repeat_penalty = atof(next());
        else if (arg == "--eos-token")                 p.eos_token_id   = atoi(next());
        else if (arg == "--stream-ctx")                p.stream_ctx     = true;
        else if (arg == "--sink-tokens")               p.sink_tokens    = atoi(next());
        else if (arg == "--window-tokens")             p.window_tokens  = atoi(next());
        else if (arg == "--no-compress")               p.no_compress    = true;
        else if (arg == "-i" || arg == "--interactive") p.interactive   = true;
        else if (arg == "-v" || arg == "--verbose")    p.verbose        = true;
        else if (arg == "-h" || arg == "--help")       { print_usage(argv[0]); exit(0); }
        else {
            fprintf(stderr, "error: unknown option: %s\n", arg.c_str());
            return false;
        }
    }

    if (p.model.empty()) {
        fprintf(stderr, "error: --model (-m) is required\n");
        return false;
    }

    return true;
}

// ---------------------------------------------------------------------------
// Build full prompt string
// ---------------------------------------------------------------------------

static std::string build_prompt(const rkllm_params & p, const std::string & user_input) {
    std::string full;
    if (!p.system_prompt.empty()) {
        full += p.system_prompt;
        full += "\n";
    }
    full += p.input_prefix;
    full += user_input;
    full += p.input_suffix;
    return full;
}

// ---------------------------------------------------------------------------
// Interactive (chat) loop
// ---------------------------------------------------------------------------

static int run_interactive(ggml_rkllm_ctx * ctx, const rkllm_params & p) {
    fprintf(stdout, "=== PowerInfer RKLLM  [%s]  interactive mode ===\n",
            p.model_b.empty() ? "single-model" : "layer-split");
    fprintf(stdout, "Type your message and press Enter.  Empty line to quit.\n\n");

    std::string line;
    while (!g_abort_requested.load()) {
        fprintf(stdout, "> ");
        fflush(stdout);

        if (!std::getline(std::cin, line) || line.empty()) break;

        std::string prompt = build_prompt(p, line);
        if (p.verbose) fprintf(stderr, "\n[prompt] %s\n", prompt.c_str());

        // Start each turn fresh (clear KV-cache) so context doesn't overflow.
        ggml_rkllm_clear_kv_cache(ctx);

        decode_stats s = run_decode(ctx, p, prompt);
        print_stats(s, p.verbose);
    }

    return 0;
}

// ---------------------------------------------------------------------------
// Single-shot inference
// ---------------------------------------------------------------------------

static int run_once(ggml_rkllm_ctx * ctx, const rkllm_params & p) {
    std::string prompt = p.prompt.empty()
        ? build_prompt(p, "Hello!")
        : (p.system_prompt.empty() ? p.prompt : build_prompt(p, p.prompt));

    if (p.verbose) fprintf(stderr, "[prompt]\n%s\n", prompt.c_str());

    decode_stats s = run_decode(ctx, p, prompt);
    print_stats(s, p.verbose);
    return 0;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char ** argv) {
    rkllm_params p;

    if (!parse_args(argc, argv, p)) {
        fprintf(stderr, "Run '%s --help' for usage.\n", argv[0]);
        return 1;
    }

    signal(SIGINT,  signal_handler);
    signal(SIGTERM, signal_handler);

    // ── Load first-half (or full) model ─────────────────────────────────────
    fprintf(stderr, "[rkllm] loading model: %s\n", p.model.c_str());

    ggml_rkllm_ctx * ctx = ggml_rkllm_init(
        p.model.c_str(),
        p.num_npu_core,
        p.n_ctx,
        p.n_predict,
        p.temp, p.top_p, p.top_k, p.repeat_penalty,
        /*callback=*/nullptr   // text streaming not used; we print from decode loop
    );

    if (!ggml_rkllm_is_loaded(ctx)) {
        fprintf(stderr,
            "[rkllm] failed to load model.  Check:\n"
            "  1. librkllmrt.so is on LD_LIBRARY_PATH\n"
            "  2. Model path exists and is a valid .rkllm file\n"
            "  3. Running on RK3588 / RK3588S hardware\n");
        return 1;
    }

    // ── Optionally load second-half model (layer-split mode) ────────────────
    if (!p.model_b.empty()) {
        fprintf(stderr, "[rkllm] loading second-half model: %s\n", p.model_b.c_str());
        if (ggml_rkllm_load_second_half(ctx, p.model_b.c_str(), p.num_npu_core) != 0) {
            fprintf(stderr, "[rkllm] failed to load second-half model – aborting\n");
            ggml_rkllm_free(ctx);
            return 1;
        }
        fprintf(stderr, "[rkllm] layer-split mode active\n");
    } else {
        fprintf(stderr, "[rkllm] single-model mode (CPU sampling only)\n");
    }

    // ── Context compression (StreamingLLM sliding window) ───────────────────
    if (p.stream_ctx) {
        RKLLMCtxCfg cfg;
        cfg.mode            = RKLLM_CTX_SLIDING;
        cfg.sink_tokens     = p.sink_tokens;
        cfg.window_tokens   = p.window_tokens;   // 0 = auto
        cfg.compress_hidden = !p.no_compress;
        ggml_rkllm_set_ctx_cfg(ctx, &cfg);

        int32_t eff_win = (p.window_tokens > 0)
            ? p.window_tokens
            : (p.n_ctx - p.sink_tokens);
        fprintf(stderr,
            "[ctx] sliding-window ON  sinks=%d  window=%d  "
            "effective_ctx=%d  hidden_compress=%s\n",
            p.sink_tokens, eff_win, p.sink_tokens + eff_win,
            p.no_compress ? "off" : "INT8");
    } else {
        // Compress hidden states in split mode even without sliding window.
        if (!p.model_b.empty() && !p.no_compress) {
            RKLLMCtxCfg cfg = ggml_rkllm_get_ctx_cfg(ctx);
            cfg.compress_hidden = true;
            ggml_rkllm_set_ctx_cfg(ctx, &cfg);
        }
    }

    g_ctx = ctx;

    int ret = p.interactive ? run_interactive(ctx, p) : run_once(ctx, p);

    ggml_rkllm_free(ctx);
    g_ctx = nullptr;
    return ret;
}
