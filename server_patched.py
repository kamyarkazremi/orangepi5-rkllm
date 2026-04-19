#!/usr/bin/env python3
"""
Lightweight Ollama-compatible API server wrapping the rkllm binary.
Exposes /api/tags and /api/chat on port 11434.
"""
import subprocess, threading, queue, time, json, os, sys, datetime, re, pty, fcntl, signal, atexit, termios
from flask import Flask, request, jsonify, Response, stream_with_context

MODEL_PATH  = os.environ.get("RKLLM_MODEL_PATH", "/srv/nvme-share/models/Qwen3-4B-1.2.0.rkllm")
MODEL_NAME  = os.environ.get("RKLLM_MODEL_NAME", "qwen3:4b")
PORT        = int(os.environ.get("RKLLM_PORT", "11434"))
RKLLM_BIN   = os.environ.get("RKLLM_BIN", "/usr/bin/rkllm")

app         = Flask(__name__)
proc        = None
pty_master  = None   # PTY master fd for writing to rkllm stdin
proc_lock   = threading.Lock()
output_q    = queue.Queue()
ready       = threading.Event()
rkllm_idle  = threading.Event()  # set when rkllm has shown "You:" (ready for next input)
needs_restart = False             # set when context corruption requires restart on next request

STRIP_RE = re.compile(r'\x1b\[[0-9;]*m|\r|<\|im_end\|>|<\|im_start\|>[a-z]*\n?')

def clean(line: str) -> str:
    return STRIP_RE.sub('', line)


def start_rkllm():
    global proc, pty_master
    print(f"[rkllm-api] Starting {RKLLM_BIN} {MODEL_PATH} via PTY", flush=True)

    # Use a PTY so rkllm behaves as in interactive mode (full prompts/markers)
    master_fd, slave_fd = pty.openpty()
    pty_master = master_fd

    # Make master non-blocking for reads
    flags = fcntl.fcntl(master_fd, fcntl.F_GETFL)
    fcntl.fcntl(master_fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

    proc = subprocess.Popen(
        [RKLLM_BIN, MODEL_PATH, "4096", "2048"],
        stdin=slave_fd,
        stdout=slave_fd,
        stderr=slave_fd,
        close_fds=True,
    )
    os.close(slave_fd)  # parent doesn't need slave end

    startup_buf = b""

    def reader():
        nonlocal startup_buf
        tail = b""  # rolling tail buffer for "You:" detection
        while True:
            try:
                chunk = os.read(master_fd, 256)
            except BlockingIOError:
                time.sleep(0.01)
                continue
            except OSError:
                # rkllm process died (PTY closed) — unblock waiting requests
                # so the next run_chat call triggers _restart_nolock()
                print("[rkllm] reader: PTY closed — rkllm_enhanced died unexpectedly", flush=True)
                global needs_restart
                needs_restart = True
                rkllm_idle.set()   # unblock rkllm_idle.wait() in run_chat
                break
            if not chunk:
                break
            if not ready.is_set():
                startup_buf += chunk
                if b"You:" in startup_buf:
                    ready.set()
                    rkllm_idle.set()
                    after = startup_buf.split(b"You:", 1)[1]
                    if after.strip():
                        output_q.put(after)
            else:
                output_q.put(chunk)
                # Detect "You:" in a rolling window to signal rkllm is idle
                tail = (tail + chunk)[-20:]
                if b"You:" in tail:
                    rkllm_idle.set()

    threading.Thread(target=reader, daemon=True).start()

    print("[rkllm-api] Waiting for model to load...", flush=True)
    ready.wait(timeout=90)
    time.sleep(0.5)
    print(f"[rkllm-api] Model ready. startup_buf tail={startup_buf[-200:]!r}", flush=True)


_REFUSAL_PREFIXES = (
    "i'm sorry, but i can't",
    "i'm sorry, but i cannot",
    "i cannot assist",
    "i can't assist",
    "i apologize, but i",
    "i'm unable to",
    "i am unable to",
    "i'm sorry, i cannot",
    "sorry, but i can't",
)

def _is_refusal(text: str) -> bool:
    t = text.lower().strip()
    return any(t.startswith(p) for p in _REFUSAL_PREFIXES)


def messages_to_prompt(messages: list) -> str:
    """Build a prompt from messages.

    On the first user turn we prepend the system message (context) so the model
    has access to transcript/notes/KB.  On subsequent turns rkllm keeps the
    conversation in its context window so we only need to send the new user text.
    """
    system = ""
    for m in messages:
        if m.get("role") == "system":
            system = m.get("content", "")
            break

    # Count how many assistant replies have already been sent (= turns completed)
    prior_turns = sum(1 for m in messages if m.get("role") == "assistant")

    # Get the latest user message
    user_msg = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            user_msg = m.get("content", "")
            break

    if system and prior_turns == 0:
        # First turn — inject context so the model can answer about the video.
        # Omit "Question:/Answer:" wrappers: they trigger verbose textbook-mode
        # in qwen2.5 small models despite brevity instructions in the system prompt.
        #
        # Hard limit: qwen2.5-vl-3b on this NPU is very slow at prefill for large
        # contexts (4096-token context window, ~50s prefill per 1000 tokens).  Cap
        # the combined prompt at ~3800 bytes to stay well under one-pass capacity
        # and deliver responses in a reasonable time.  The limit is applied by
        # truncating only the "Reference material:" / content section — the system
        # instructions and user question are always preserved in full.
        MAX_TOTAL = 3800
        prompt = f"{system}\n\n{user_msg}"
        if len(prompt.encode('utf-8')) > MAX_TOTAL:
            # Split on "Reference material:" to find where content starts.
            split_marker = "Reference material:"
            ref_idx = system.find(split_marker)
            if ref_idx >= 0:
                instructions = system[:ref_idx + len(split_marker)]
                content      = system[ref_idx + len(split_marker):]
                # How many bytes are left for content after reserving instructions + user_msg?
                reserved = len(instructions.encode()) + len(user_msg.encode()) + 20  # "\n\n" + buffer
                budget   = max(0, MAX_TOTAL - reserved)
                content_trunc = content.encode('utf-8')[:budget].decode('utf-8', errors='ignore')
                print(f"[messages_to_prompt] truncated content {len(content)}->{len(content_trunc)} chars", flush=True)
                prompt = f"{instructions}{content_trunc}\n\n{user_msg}"
            else:
                # No recognisable split point — truncate the whole system message.
                budget = MAX_TOTAL - len(user_msg.encode()) - 10
                sys_trunc = system.encode('utf-8')[:budget].decode('utf-8', errors='ignore')
                prompt = f"{sys_trunc}\n\n{user_msg}"
        return prompt

    if system and prior_turns > 0:
        # On follow-up turns rkllm holds conversation in its PTY context window,
        # but if it was restarted the context is gone.  Re-inject a short reminder
        # (just the first 600 chars of the system prompt) so the model stays on-topic.
        reminder = system[:600].rstrip()
        return f"[Context: {reminder}…] {user_msg}"

    return user_msg


def _kill_rkllm():
    """Kill the rkllm subprocess, with SIGKILL fallback. Safe to call at any time."""
    global proc, pty_master
    if pty_master is not None:
        try:
            os.close(pty_master)
        except OSError:
            pass
        pty_master = None
    if proc and proc.poll() is None:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print("[rkllm] SIGTERM timeout — sending SIGKILL", flush=True)
            proc.kill()
            proc.wait(timeout=3)
        except Exception as e:
            print(f"[rkllm] kill error: {e}", flush=True)
    proc = None


def _interrupt_or_restart():
    """After a successful response, interrupt rkllm generation via Ctrl-C so it returns
    to the 'You:' idle prompt quickly (~1s) instead of reloading the model (~30s).
    Falls back to a full restart if Ctrl-C doesn't work within 8 seconds."""
    global pty_master
    if pty_master is None:
        _restart_nolock()
        return
    try:
        os.write(pty_master, b'\x03')  # Ctrl-C → interrupt current generation
        print("[rkllm] sent Ctrl-C to interrupt generation", flush=True)
    except OSError as e:
        print(f"[rkllm] Ctrl-C failed: {e} — falling back to restart", flush=True)
        _restart_nolock()
        return
    # Wait up to 8 s for rkllm to return to the 'You:' prompt after Ctrl-C.
    if rkllm_idle.wait(timeout=8):
        print("[rkllm] interrupted OK — back to idle", flush=True)
    else:
        print("[rkllm] Ctrl-C didn't return to idle in 8s — doing full restart", flush=True)
        _restart_nolock()


def _restart_nolock():
    """Kill rkllm and restart. Caller must NOT hold proc_lock (or be fine holding it)."""
    print("[restart] terminating rkllm for restart", flush=True)
    _kill_rkllm()
    ready.clear()
    rkllm_idle.clear()
    while True:
        try:
            output_q.get_nowait()
        except queue.Empty:
            break
    start_rkllm()


def _write_to_rkllm(master_fd: int, data: bytes):
    """Write prompt bytes to the PTY master.

    Linux PTY canonical mode limits a line to MAX_CANON=4096 bytes.  For prompts
    longer than that we temporarily disable ICANON on the line discipline, write
    in 512-byte chunks (paced so rkllm can drain the ring buffer), then restore
    the original mode.  This is safe to do while rkllm is blocked in its read()
    loop waiting for input — the mode change only affects buffering, not ISIG, so
    Ctrl-C interrupts still work for the response.
    """
    if len(data) <= 3800:
        # Short enough for canonical mode — write directly.
        os.write(master_fd, data)
        return

    # Long prompt — bypass canonical 4096-byte limit.
    print(f"[write] long prompt ({len(data)} bytes) — bypassing canonical limit", flush=True)
    try:
        attrs_orig = termios.tcgetattr(master_fd)
        attrs_raw = list(attrs_orig)
        attrs_raw[3] &= ~termios.ICANON   # clear ICANON, keep ECHO+ISIG
        attrs_raw[6] = list(attrs_raw[6])
        attrs_raw[6][termios.VMIN]  = 1
        attrs_raw[6][termios.VTIME] = 0
        termios.tcsetattr(master_fd, termios.TCSANOW, attrs_raw)
    except Exception as e:
        print(f"[write] tcsetattr failed ({e}) — writing directly", flush=True)
        os.write(master_fd, data)
        return

    try:
        view = memoryview(data)
        offset = 0
        while offset < len(data):
            try:
                n = os.write(master_fd, bytes(view[offset:offset + 512]))
                offset += n
                if offset < len(data):
                    time.sleep(0.01)   # pace at ~50 KB/s so ring buffer doesn't fill
            except BlockingIOError:
                time.sleep(0.02)
        print(f"[write] wrote {len(data)} bytes OK", flush=True)
    finally:
        try:
            termios.tcsetattr(master_fd, termios.TCSANOW, attrs_orig)
        except Exception as e:
            print(f"[write] restore tcsetattr failed: {e}", flush=True)


def run_chat(prompt: str):
    """Send prompt to rkllm subprocess and yield response text chunks."""
    global needs_restart

    # If context corruption was flagged, restart now (synchronous, within proc_lock).
    if needs_restart:
        needs_restart = False
        print("[drain] needs_restart — restarting rkllm before next prompt", flush=True)
        _restart_nolock()
    elif not ready.is_set():
        # Background restart from previous response is in progress; wait for it.
        print("[drain] waiting for background restart...", flush=True)
        if not ready.wait(timeout=90):
            print("[drain] background restart timeout — giving up", flush=True)
            return

    # Wait for rkllm to show "You:" (idle prompt) before sending next input.
    # rkllm_idle is set by the reader thread whenever it sees "You:" in output.
    if not rkllm_idle.is_set():
        if not rkllm_idle.wait(timeout=120):
            print("[drain] timeout waiting for rkllm idle — restarting rkllm", flush=True)
            _restart_nolock()
            if not rkllm_idle.wait(timeout=90):
                print("[drain] restart timeout — giving up", flush=True)
                return
    rkllm_idle.clear()  # reset — will be set again after next response finishes
    # Drain any buffered output after "You:"
    drain_buf = b""
    while True:
        try:
            drain_buf += output_q.get_nowait()
        except queue.Empty:
            break
    print(f"[drain] ready, drained {len(drain_buf)} extra bytes", flush=True)

    # PTY canonical mode processes input line-by-line — each \n in the prompt
    # would be submitted as a separate query to rkllm, causing runaway generation.
    # Collapse all internal newlines to spaces so the whole prompt arrives as one line.
    pty_prompt = " ".join(line for line in prompt.splitlines() if line.strip()) or prompt.replace("\n", " ")
    encoded = (pty_prompt + "\n").encode('utf-8')
    print(f"[run_chat] sending ({len(encoded)} bytes): {pty_prompt[:80]!r}", flush=True)
    _write_to_rkllm(pty_master, encoded)

    # Qwen3 with rkllm uses multiple response-start formats depending on context:
    # A) <｜End of turn｜> → simple text or structured wrapper → response
    # B) <｜End of the conversation｜> → <|Start of the response|> → response
    # C) <|Start of the response|> directly (ASCII pipes, not U+FF5C)
    # D) </think> → answer  (ChatML / rkllm_enhanced binary with n_keep sliding window)
    RESPONSE_MARKERS = [
        b"<\xef\xbd\x9cEnd of turn\xef\xbd\x9c>",               # A: U+FF5C
        b"<\xef\xbd\x9cEnd of the conversation\xef\xbd\x9c>",   # B: U+FF5C
        b"<|Start of the response|>",                            # C: ASCII pipes
        b"</think>",                                              # D: ChatML think-end
    ]

    def find_response_marker(buf):
        best = (-1, -1)
        for m in RESPONSE_MARKERS:
            idx = buf.find(m)
            if idx >= 0 and (best[0] < 0 or idx < best[0]):
                best = (idx, idx + len(m))
        return best

    END_MARKERS     = [
        "<｜User｜>:".encode('utf-8'),       # rkllm interactive prompt style (with colon)
        "<｜User｜>".encode('utf-8'),        # model-generated separator (no colon, qwen2.5-vl)
        "<｜Assistant｜>:".encode('utf-8'),  # qwen3 turn separator (with colon)
        "<｜Assistant｜>".encode('utf-8'),   # qwen2.5 turn separator (no colon) — stops loops
        b"You:",
        b"<|End of the response|>",          # ASCII pipes variant
        b"<|im_end|>",                       # ChatML response-end (rkllm_enhanced / n_keep mode)
        b"Human:",
        b"\nHuman:",
        RESPONSE_MARKERS[0],                 # second <｜End of turn｜> = new turn
    ]

    # Structured format: actual answer is after <｜Content｜>:
    CONTENT_MARKER = b"<\xef\xbd\x9cContent\xef\xbd\x9c>: "
    # ASCII response body marker (used after <｜End of the conversation｜>)
    RESPONSE_BODY_MARKER = b"<|Start of the response|>"
    # Qwen3 rkllm format: <｜Assistant｜>:\r\n[response]\r\n<｜User｜>:
    ASSISTANT_MARKER = b"<\xef\xbd\x9cAssistant\xef\xbd\x9c>:"

    buf             = b""
    # Phases: 'wait_llm' → 'wait_nl' → 'skip_think' [→ 'skip_template'] → 'collect'
    phase           = 'wait_llm'
    last_chunk_time = time.time()
    timeout_at      = time.time() + 600

    while time.time() < timeout_at:
        # Fast exit if subprocess was killed externally (e.g. user abort)
        if proc is not None and proc.poll() is not None:
            print("[run_chat] rkllm process exited, aborting run_chat", flush=True)
            needs_restart = True
            return
        try:
            chunk = output_q.get(timeout=0.5)
        except queue.Empty:
            if phase == 'collect' and (time.time() - last_chunk_time) > 30.0:
                break
            continue

        buf += chunk
        last_chunk_time = time.time()

        if phase == 'wait_llm':
            if b"LLM:" in buf:
                buf   = buf.split(b"LLM:", 1)[1]
                phase = 'wait_nl'

        if phase == 'wait_nl':
            # qwen3 uses a header-only first line (" :\r\n"), thinking follows.
            # qwen2.5/non-thinking models put the response directly on the LLM: line.
            # Also check end-markers here: short responses may never emit a \n.
            _end_found = False
            for _end in END_MARKERS:
                if _end in buf:
                    part = buf.split(_end, 1)[0].strip(b" \r\n")
                    text = clean(part.decode('utf-8', errors='replace')).strip()
                    if text:
                        print(f"[run_chat] end marker in wait_nl: {text[:60]!r}", flush=True)
                        yield text
                    _end_found = True
                    break
            if _end_found:
                return
            if b"\n" in buf:
                nl_idx = buf.index(b"\n")
                first_line = buf[:nl_idx].strip(b" \r\n")
                if first_line in (b"", b":", b"<think>"):
                    # Empty/header-only/ChatML-think-start → skip thinking section
                    buf   = buf[nl_idx + 1:]
                    phase = 'skip_think'
                else:
                    # Response starts on same line as LLM: (non-thinking model)
                    buf   = buf.lstrip(b" ")
                    phase = 'collect'

        if phase == 'skip_think':
            # Wait for <｜End of turn｜> which marks end of thinking.
            # Fallback: if rkllm ends without the marker (context overload), return empty.
            m_start, m_end = find_response_marker(buf)
            if m_start >= 0:
                buf   = buf[m_end:]
                buf   = buf.lstrip(b" \r\n")
                phase = 'skip_template'
                print(f"[run_chat] response_marker found, entering skip_template", flush=True)
                # Fall through immediately to skip_template processing below.
            else:
                for end in END_MARKERS:
                    if end in buf:
                        part = buf.split(end, 1)[0]
                        # Model may have included thinking tokens with <｜Assistant｜>: markers.
                        # Use only what follows the LAST <｜Assistant｜>: to avoid returning thinking content.
                        asst_mark = b"<\xef\xbd\x9cAssistant\xef\xbd\x9c>:"
                        last_asst = part.rfind(asst_mark)
                        if last_asst >= 0:
                            part = part[last_asst + len(asst_mark):]
                        text = clean(part.decode('utf-8', errors='replace')).strip()
                        if text:
                            print(f"[run_chat] direct response: {text[:60]!r}", flush=True)
                            yield text
                        else:
                            print("[run_chat] no content before end marker — flagging restart", flush=True)
                            needs_restart = True
                        return

        if phase == 'skip_template':
            # After any response-start marker, three sub-formats are possible:
            # 1. Simple: regular text starts immediately
            # 2. U+FF5C structured: <｜Assistant｜>...<｜Content｜>: [response]
            # 3. ASCII structured: <|Start of the response|> [response]
            # Need ≥4 bytes to distinguish template tokens from regular text.
            if len(buf) >= 4:
                if not (buf.startswith(b"<\xef\xbd\x9c") or buf.startswith(b"<|")):
                    # No template prefix — response starts immediately
                    phase = 'collect'
                else:
                    # Template prefix — look for content-start markers
                    for cm in (CONTENT_MARKER, RESPONSE_BODY_MARKER, ASSISTANT_MARKER):
                        idx = buf.find(cm)
                        if idx >= 0:
                            buf   = buf[idx + len(cm):]
                            buf   = buf.lstrip(b" \r\n")
                            phase = 'collect'
                            break
            # Check for premature end markers while waiting in skip_template.
            if phase == 'skip_template':
                for end in END_MARKERS:
                    if end in buf:
                        print(f"[run_chat] end before content in skip_template — end={end!r} buf={buf[:100]!r}", flush=True)
                        needs_restart = True
                        return

        if phase == 'collect':
            done = False
            for marker in END_MARKERS:
                if marker in buf:
                    part = buf.split(marker, 1)[0]
                    text = clean(part.decode('utf-8', errors='replace')).strip()
                    if text:
                        yield text
                    done = True
                    break
            if done:
                break
            # Keep a 24-byte safety tail so end markers can't be split across yields
            # Longest END_MARKER is ~20 bytes (<｜User｜>:) so 24 bytes is sufficient.
            if len(buf) > 48:
                safe = buf[:-24]
                text = clean(safe.decode('utf-8', errors='replace'))
                if text:
                    yield text
                buf = buf[-24:]


@app.post("/api/reset")
def api_reset():
    """Restart rkllm to clear accumulated conversation context."""
    with proc_lock:
        _kill_rkllm()
        ready.clear()
        rkllm_idle.clear()
        while True:
            try:
                output_q.get_nowait()
            except queue.Empty:
                break
        start_rkllm()
    return jsonify({"ok": True, "status": "rkllm restarted"})


@app.get("/")
def index():
    return jsonify({"message": "rkllm-api — Ollama-compatible", "model": MODEL_NAME})


@app.get("/api/tags")
def api_tags():
    size = os.path.getsize(MODEL_PATH) if os.path.exists(MODEL_PATH) else 0
    return jsonify({"models": [{
        "name":        MODEL_NAME,
        "model":       MODEL_NAME,
        "modified_at": datetime.datetime.utcnow().isoformat() + "Z",
        "size":        size,
        "details":     {"family": "qwen", "format": "rkllm", "parameter_size": "4B"},
    }]})


@app.post("/api/chat")
def api_chat():
    if not ready.is_set():
        return jsonify({"error": "Model not ready"}), 503

    data     = request.get_json(force=True)
    messages = data.get("messages", [])
    stream   = data.get("stream", True)
    prompt   = messages_to_prompt(messages)
    now      = datetime.datetime.utcnow().isoformat() + "Z"

    if stream:
        def generate():
            try:
                with proc_lock:
                    tokens = []
                    refusal_checked = False
                    for token in run_chat(prompt):
                        tokens.append(token)
                        # Check for refusal on the first ~60 chars
                        if not refusal_checked and sum(len(t) for t in tokens) >= 40:
                            refusal_checked = True
                            so_far = "".join(tokens)
                            if _is_refusal(so_far):
                                print(f"[rkllm-api] refusal detected — restarting", flush=True)
                                _restart_nolock()
                                # Emit a neutral error token so the UI shows something
                                yield json.dumps({"model": MODEL_NAME, "created_at": now,
                                                  "message": {"role": "assistant",
                                                               "content": "[Model refused. Context cleared — please re-inject and try again.]"},
                                                  "done": False}) + "\n"
                                break
                        chunk = {"model": MODEL_NAME, "created_at": now,
                                 "message": {"role": "assistant", "content": token},
                                 "done": False}
                        yield json.dumps(chunk) + "\n"
                    else:
                        # Normal completion — interrupt back to idle
                        rkllm_idle.clear()
                        threading.Thread(target=_interrupt_or_restart, daemon=True).start()
                yield json.dumps({"model": MODEL_NAME, "created_at": now,
                                  "message": {"role": "assistant", "content": ""},
                                  "done": True, "done_reason": "stop"}) + "\n"
            except Exception as e:
                print(f"[rkllm-api] generate() error: {e}", flush=True)
                yield json.dumps({"error": str(e)}) + "\n"
        return Response(stream_with_context(generate()), content_type="application/x-ndjson")
    else:
        with proc_lock:
            full = "".join(run_chat(prompt))
            rkllm_idle.clear()
            threading.Thread(target=_interrupt_or_restart, daemon=True).start()
        return jsonify({"model": MODEL_NAME, "created_at": now,
                        "message": {"role": "assistant", "content": full},
                        "done": True, "done_reason": "stop"})


@app.post("/api/generate")
def api_generate():
    if not ready.is_set():
        return jsonify({"error": "Model not ready"}), 503

    data   = request.get_json(force=True)
    prompt = data.get("prompt", "")
    stream = data.get("stream", True)
    now    = datetime.datetime.utcnow().isoformat() + "Z"

    if stream:
        def generate():
            with proc_lock:
                for token in run_chat(prompt):
                    yield json.dumps({"model": MODEL_NAME, "created_at": now,
                                      "response": token, "done": False}) + "\n"
                rkllm_idle.clear()
                threading.Thread(target=_interrupt_or_restart, daemon=True).start()
            yield json.dumps({"model": MODEL_NAME, "created_at": now,
                              "response": "", "done": True}) + "\n"
        return Response(stream_with_context(generate()), content_type="application/x-ndjson")
    else:
        with proc_lock:
            full = "".join(run_chat(prompt))
            rkllm_idle.clear()
            threading.Thread(target=_interrupt_or_restart, daemon=True).start()
        return jsonify({"model": MODEL_NAME, "created_at": now,
                        "response": full, "done": True})


def _shutdown(signum=None, frame=None):
    print(f"[rkllm-api] shutting down (signal {signum})", flush=True)
    _kill_rkllm()
    sys.exit(0)


if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found: {MODEL_PATH}", file=sys.stderr)
        sys.exit(1)
    atexit.register(_kill_rkllm)
    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)
    start_rkllm()
    print(f"[rkllm-api] Serving on http://0.0.0.0:{PORT}", flush=True)
    app.run(host="0.0.0.0", port=PORT, threaded=True, debug=False)
