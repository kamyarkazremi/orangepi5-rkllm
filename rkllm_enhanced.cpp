// rkllm_enhanced.cpp  — drop-in replacement for /usr/bin/rkllm
//
// Improvements over stock binary:
//   n_keep=4         : attention-sink sliding window — constant KV-cache RAM
//                      regardless of conversation length (SDK handles eviction)
//   embed_flash=1    : load embedding table from NVMe, not DRAM (~500 MB saved)
//   cpus_mask=0xF0   : pin NPU worker threads to Cortex-A76 big cores (CPU4-7)
//
// Protocol (identical to stock binary — server.py compatible):
//   startup : prints init lines then "You: " to signal readiness
//   input   : one prompt per line from stdin (PTY)
//   output  : "\nLLM: " + response text, then "\nYou: " when done
//
// Usage (same args as stock binary):
//   rkllm_enhanced <model.rkllm> [max_context_len] [max_new_tokens]

#include <cstddef>   // size_t — must come before rkllm.h (SDK header omits it)
#include <rkllm.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <csignal>
#include <atomic>

static LLMHandle         g_handle   = nullptr;
static std::atomic<bool> g_running  {false};
static std::atomic<bool> g_first_tok{true};

// ---------------------------------------------------------------------------
// Callback: stream text tokens to stdout
// ---------------------------------------------------------------------------
static void llm_cb(RKLLMResult* res, void* /*ud*/, LLMCallState state)
{
    switch (state) {
    case RKLLM_RUN_NORMAL:
    case RKLLM_RUN_WAITING:
        if (res && res->text && res->text[0]) {
            if (g_first_tok.exchange(false)) {
                // Emit the "LLM: " marker server.py waits for
                fputs("\nLLM: ", stdout);
            }
            fputs(res->text, stdout);
            fflush(stdout);
        }
        break;
    case RKLLM_RUN_FINISH:
        g_running.store(false);
        break;
    case RKLLM_RUN_ERROR:
        fputs("\n[rkllm error]\n", stderr);
        g_running.store(false);
        break;
    }
}

// ---------------------------------------------------------------------------
// SIGINT: abort current generation, re-print idle prompt
// ---------------------------------------------------------------------------
static void sigint_handler(int /*sig*/)
{
    if (g_handle && g_running.load())
        rkllm_abort(g_handle);
    fputs("\nYou: ", stdout);
    fflush(stdout);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    if (argc < 2) {
        fputs("Usage: rkllm_enhanced <model.rkllm> [max_ctx] [max_new]\n", stderr);
        return 1;
    }

    const char* model_path = argv[1];
    int max_ctx = (argc > 2) ? atoi(argv[2]) : 4096;
    int max_new = (argc > 3) ? atoi(argv[3]) : 2048;

    // ── Build parameters ──────────────────────────────────────────────────────
    RKLLMParam p = rkllm_createDefaultParam();
    p.model_path       = model_path;
    p.max_context_len  = max_ctx;
    p.max_new_tokens   = max_new;

    // Attention-sink sliding window:
    //   SDK keeps the first n_keep KV slots (attention sinks) when the cache
    //   fills.  Older middle tokens are evicted.  Result: constant DRAM usage
    //   at (n_keep + window_size) * kv_bytes regardless of how long you chat.
    p.n_keep           = 4;

    // Sampling
    p.temperature      = 0.7f;
    p.top_k            = 40;
    p.top_p            = 0.9f;
    p.repeat_penalty   = 1.1f;
    p.mirostat         = 0;

    // Keep special tokens visible so server.py can parse Qwen3 thinking markers
    p.skip_special_token = false;

    // embed_flash: serve the embedding table from NVMe/flash instead of DRAM.
    //   The NPU prefetches embedding rows; NVMe latency is largely hidden.
    //   Saves ~(vocab_size * embed_dim * 2) bytes of DRAM — ~500 MB for Qwen3-4B.
    p.extend_param.embed_flash = 1;

    // Cortex-A76 big-core affinity: CPU4-7 on RK3588 (CPU4=0x10..CPU7=0x80).
    //   Keeps NPU worker threads off the efficiency A55 cores during inference.
    p.extend_param.enabled_cpus_num  = 4;
    p.extend_param.enabled_cpus_mask = 0xF0u;

    // ── Load model ────────────────────────────────────────────────────────────
    fprintf(stdout,
        "I rkllm_enhanced: loading %s\n"
        "I rkllm_enhanced: ctx=%d  new=%d  n_keep=%d  embed_flash=%d  cpus=0x%02X\n",
        model_path, max_ctx, max_new,
        p.n_keep,
        (int)p.extend_param.embed_flash,
        (unsigned)p.extend_param.enabled_cpus_mask);
    fflush(stdout);

    int ret = rkllm_init(&g_handle, &p, llm_cb);
    if (ret != 0) {
        fputs("rkllm init failed\n", stderr);
        return 1;
    }

    // Override chat template to match the stock rkllm binary format.
    // server.py's response parser is tuned to the <｜User｜>/<｜Assistant｜>
    // delimiters (U+FF5C wide pipes) which produce <｜End of turn｜> thinking
    // markers that server.py uses for its skip_think phase detection.
    // Without this, the model uses ChatML (<|im_start|>) and server.py can't
    // find the response boundary markers.
    // U+FF5C (FULLWIDTH VERTICAL LINE) = 0xEF BD 9C in UTF-8.
    // String literal splitting prevents the C parser merging \x9c + "A…"
    // into a single out-of-range escape (0x9CA).
    rkllm_set_chat_template(g_handle,
        "",                                              // system_prompt
        "<\xef\xbd\x9c" "User\xef\xbd\x9c>",           // <｜User｜>
        "<\xef\xbd\x9c" "Assistant\xef\xbd\x9c>");      // <｜Assistant｜>

    signal(SIGINT, sigint_handler);

    // server.py waits for "You:" in startup output before marking model ready
    fputs("\nYou: ", stdout);
    fflush(stdout);

    // ── Inference loop ────────────────────────────────────────────────────────
    static char prompt[16384];
    int turn = 0;

    while (fgets(prompt, (int)sizeof(prompt), stdin)) {
        // Strip trailing CR/LF
        size_t len = strlen(prompt);
        while (len > 0 && (prompt[len-1] == '\n' || prompt[len-1] == '\r'))
            prompt[--len] = '\0';
        if (len == 0)
            continue;

        g_first_tok.store(true);
        g_running.store(true);

        RKLLMInput inp;
        inp.input_type   = RKLLM_INPUT_PROMPT;
        inp.prompt_input = prompt;

        RKLLMInferParam ip;
        memset(&ip, 0, sizeof(ip));
        ip.mode         = RKLLM_INFER_GENERATE;
        // keep_history=0 on the very first turn (fresh KV cache),
        // keep_history=1 on subsequent turns (extend existing context).
        ip.keep_history = (turn > 0) ? 1 : 0;

        rkllm_run(g_handle, &inp, &ip, nullptr);
        g_running.store(false);
        turn++;

        // Signal idle to server.py — it looks for "You:" in a rolling tail
        fputs("\nYou: ", stdout);
        fflush(stdout);
    }

    rkllm_destroy(g_handle);
    return 0;
}
