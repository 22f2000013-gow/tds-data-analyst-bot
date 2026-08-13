"""Data-analyst Telegram bot - TDS Project 1.

An LLM agent that answers data-analysis questions sent over Telegram and
replies with exactly one JSON object, shaped as the question asks:

    {"answer": <shaped as asked>, "log_url": "https://<host>/run.jsonl"}

Architecture:
  - FastAPI app serves /health and /run.jsonl (the public agent log).
  - A background thread long-polls Telegram getUpdates.
  - Each incoming message runs an agentic loop (OpenAI-compatible chat with a
    run_python tool) until the model produces the final JSON answer.
  - A keep-warm thread pings our own public URL so a free host never idles out.
"""

import contextlib
import io
import json
import os
import re
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import requests
from fastapi import FastAPI
from fastapi.responses import FileResponse, PlainTextResponse

# ---------------------------------------------------------------- config
BOT_TOKEN = os.environ.get("BOT_TOKEN", "")
AIPIPE_TOKEN = os.environ.get("AIPIPE_TOKEN", "")
MODEL = os.environ.get("MODEL", "gpt-5.6-terra")
MODEL_BASE_URL = os.environ.get("MODEL_BASE_URL", "https://aipipe.org/openai/v1").rstrip("/")
BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000").rstrip("/")
LOG_PATH = os.environ.get("LOG_PATH", "/tmp/run.jsonl")
LOG_URL = os.environ.get("LOG_URL", f"{BASE_URL}/run.jsonl")
TG_API = f"https://api.telegram.org/bot{BOT_TOKEN}"

MAX_AGENT_STEPS = int(os.environ.get("MAX_AGENT_STEPS", "8"))
PY_TIMEOUT = int(os.environ.get("PY_TIMEOUT", "60"))       # per run_python call
ANSWER_BUDGET = int(os.environ.get("ANSWER_BUDGET", "150"))  # sec before forcing an answer
# 1 = always reply {"answer":..., "log_url":...}; 0 = mirror the shape the message asks for
FORCE_WRAP = os.environ.get("FORCE_WRAP", "0") == "1"

_log_lock = threading.Lock()
_hist_lock = threading.Lock()
_histories: dict[int, list[dict]] = {}  # chat_id -> chat-completion messages


# ---------------------------------------------------------------- logging
def log_event(**fields):
    """Append one JSON object per line to the public run log."""
    fields["ts"] = datetime.now(timezone.utc).isoformat()
    line = json.dumps(fields, ensure_ascii=False, default=str)
    try:
        with _log_lock:
            with open(LOG_PATH, "a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception:
        pass


# ---------------------------------------------------------------- tools
def run_python(code: str) -> str:
    """Execute Python code, return captured stdout/stderr (or the traceback)."""
    out = io.StringIO()

    def target():
        env = {"__name__": "__main__"}
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
                exec(code, env)
        except Exception:
            out.write("\n" + traceback.format_exc(limit=4))

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(PY_TIMEOUT)
    if t.is_alive():
        return f"ERROR: code timed out after {PY_TIMEOUT}s"
    text = out.getvalue()
    return text[-8000:] if text.strip() else "(no output - remember to print() what you need)"


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": (
                "Run Python code on the server and get its printed output. "
                "pandas, numpy, requests, bs4, lxml, openpyxl are installed and the "
                "network is available (download public datasets with requests). "
                "Always print() what you need to see."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python source to execute"}
                },
                "required": ["code"],
            },
        },
    }
]

SYSTEM_PROMPT = """You are an expert data-analyst agent answering questions sent to a Telegram bot.

Rules:
1. Answer the user's LATEST message. Earlier messages are context for multi-turn tasks.
2. Data may be embedded inline in the message, or the message may reference a public
   dataset (MOSPI, data.gov.in, RBI, census, etc.). Use the run_python tool to fetch and
   compute - never guess a number you could compute. If a download fails, retry with a
   different source/URL; only fall back to well-established published knowledge as a last
   resort.
3. The message spells out the exact JSON shape it wants, e.g.
   {"answer": {"state": "<state name>"}, "log_url": "<url>"}  or  {"state": "<state name>"}.
4. Your FINAL message must be ONLY that JSON object - no prose, no markdown fences, no
   explanation. Reproduce the requested key names, nesting and types EXACTLY (numbers as
   numbers unless a string is asked for; lists in the order asked for).
5. If the shape includes "log_url", put the literal string "LOG_URL" there - the harness
   substitutes the real URL before sending.
6. If a message is only setup/context ("I'll send the data next"), reply immediately with
   the minimal valid JSON object of the requested shape (or {"answer": "ok"}), no tools.
7. Round exactly as instructed. Never invent extra keys.
8. Be efficient: you have a limited number of tool calls and a wall-clock budget.
"""


# ---------------------------------------------------------------- llm
ROUTES = [
    (os.environ.get("OPENAI_API_KEY", ""), "https://api.openai.com/v1",        MODEL),
    (AIPIPE_TOKEN,                         "https://aipipe.org/openai/v1",     MODEL),
    (AIPIPE_TOKEN,                         "https://aipipe.org/openrouter/v1", "openai/" + MODEL),
]


def chat_completion(messages, use_tools=True):
    body = {"messages": messages}
    if use_tools:
        body["tools"] = TOOLS
    last = None
    for token, base, model in ROUTES:
        if not token:
            continue
        body["model"] = model
        for attempt in range(3):
            try:
                r = requests.post(
                    f"{base}/chat/completions",
                    headers={"Authorization": f"Bearer {token}"},
                    json=body,
                    timeout=180,
                )
                if r.status_code == 429:
                    log_event(event="rate_limited", base=base, attempt=attempt, body=r.text[:300])
                    time.sleep(5 * 2 ** attempt)
                    continue
                if r.status_code >= 400:
                    log_event(event="api_error", base=base, status=r.status_code, body=r.text[:300])
                    r.raise_for_status()
                return r.json()["choices"][0]["message"]
            except Exception as e:
                last = e
                time.sleep(2)
    raise last or RuntimeError("no usable route")


# ---------------------------------------------------------------- json shaping
def extract_json(text: str):
    """Pull the first balanced JSON object out of model text."""
    if not text:
        return None
    text = re.sub(r"```(?:json)?", "", text).strip()
    start = text.find("{")
    if start == -1:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        c = text[i]
        if esc:
            esc = False
            continue
        if c == "\\":
            esc = True
            continue
        if c == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def wants_wrapper(question: str) -> bool:
    """Does the question ask for the {"answer":..., "log_url":...} envelope?

    The graded messages spell out their own shape. If the message mentions
    log_url we send the two-key envelope; otherwise we mirror the bare shape it
    asked for (the public grader compares the whole reply object by equality).
    """
    if FORCE_WRAP:
        return True
    return "log_url" in (question or "").lower()


def shape_reply(obj, question: str) -> dict:
    """Coerce the model's object into exactly the shape the message asked for."""
    if not isinstance(obj, dict):
        obj = {"answer": obj}
    keys = set(obj.keys())

    if wants_wrapper(question):
        if "answer" not in obj:
            obj = {"answer": obj}          # model returned the bare shape
        return {"answer": obj["answer"], "log_url": LOG_URL}

    # bare shape requested: unwrap if the model over-wrapped it
    if "answer" in keys and keys <= {"answer", "log_url"}:
        inner = obj["answer"]
        return inner if isinstance(inner, dict) else {"answer": inner}
    obj.pop("log_url", None)
    return obj


# ---------------------------------------------------------------- agent
def solve(chat_id: int, question: str) -> str:
    """Run the agent loop; return the final JSON reply text."""
    with _hist_lock:
        history = _histories.setdefault(chat_id, [])
        history.append({"role": "user", "content": question})
        del history[:-20]                       # keep the last 20 turns
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + list(history)

    log_event(event="question", chat_id=chat_id, text=question)

    final_text, deadline = None, time.time() + ANSWER_BUDGET
    for step in range(MAX_AGENT_STEPS):
        out_of_time = time.time() > deadline or step == MAX_AGENT_STEPS - 1
        if out_of_time:
            messages.append({
                "role": "user",
                "content": "Time is up. Reply NOW with ONLY your best final JSON object.",
            })
        try:
            msg = chat_completion(messages, use_tools=not out_of_time)
        except Exception as e:
            log_event(event="llm_error", chat_id=chat_id, step=step, error=str(e))
            time.sleep(3)
            try:
                msg = chat_completion(messages, use_tools=False)
            except Exception as e2:
                log_event(event="llm_error_final", chat_id=chat_id, error=str(e2))
                break

        tool_calls = msg.get("tool_calls")
        if tool_calls and not out_of_time:
            messages.append(msg)
            for tc in tool_calls:
                try:
                    code = json.loads(tc["function"]["arguments"]).get("code", "")
                except json.JSONDecodeError:
                    code = tc["function"]["arguments"]
                log_event(event="tool_call", chat_id=chat_id, step=step, code=code[:4000])
                output = run_python(code)
                log_event(event="tool_result", chat_id=chat_id, step=step, output=output[:4000])
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": output})
            continue

        final_text = msg.get("content") or ""
        break

    obj = extract_json(final_text)
    if obj is None:
        obj = {"answer": (final_text or "llm unavailable").strip()[:1000]}
    reply = json.dumps(shape_reply(obj, question), ensure_ascii=False)

    with _hist_lock:
        _histories.setdefault(chat_id, []).append({"role": "assistant", "content": reply})
    log_event(event="answer", chat_id=chat_id, reply=reply)
    return reply


# ---------------------------------------------------------------- telegram
def tg(method, **params):
    try:
        return requests.post(f"{TG_API}/{method}", json=params, timeout=65).json()
    except Exception as e:
        log_event(event="tg_error", method=method, error=str(e))
        return {}


def handle_update(upd):
    msg = upd.get("message") or upd.get("edited_message")
    if not msg:
        return
    text = (msg.get("text") or msg.get("caption") or "").strip()
    chat_id = msg["chat"]["id"]
    if not text:
        return
    if text.startswith("/start") or text.startswith("/help"):
        tg("sendMessage", chat_id=chat_id,
           text=json.dumps({"answer": "ready", "log_url": LOG_URL}))
        return
    try:
        reply = solve(chat_id, text)
    except Exception:
        log_event(event="agent_crash", chat_id=chat_id, error=traceback.format_exc())
        reply = json.dumps({"answer": "internal error", "log_url": LOG_URL})
    tg("sendMessage", chat_id=chat_id, text=reply[:4000])


def poll_loop():
    log_event(event="startup", base_url=BASE_URL, model=MODEL, log_url=LOG_URL)
    tg("deleteWebhook", drop_pending_updates=False)  # getUpdates fails if a webhook is set
    me = tg("getMe").get("result", {})
    log_event(event="identity", username=me.get("username"))
    offset = 0
    pool = ThreadPoolExecutor(max_workers=8)
    while True:
        try:
            resp = requests.get(
                f"{TG_API}/getUpdates",
                params={"offset": offset, "timeout": 50},
                timeout=65,
            ).json()
            for upd in resp.get("result", []):
                offset = upd["update_id"] + 1
                pool.submit(handle_update, upd)
        except Exception as e:
            log_event(event="poll_error", error=str(e))
            time.sleep(5)


def keepwarm_loop():
    """Ping our own public URL so a free host never spins down."""
    while True:
        time.sleep(480)
        try:
            requests.get(f"{BASE_URL}/health", timeout=30)
        except Exception:
            pass


# ---------------------------------------------------------------- web app
@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.path.exists(LOG_PATH):
        log_event(event="log_created")
    threading.Thread(target=poll_loop, daemon=True).start()
    if BASE_URL.startswith("https://"):
        threading.Thread(target=keepwarm_loop, daemon=True).start()
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    try:
        chat_completion([{"role": "user", "content": "say ok"}], use_tools=False)
        llm = "ok"
    except Exception as e:
        llm = str(e)[:200]
    return {"ok": True, "model": MODEL, "llm": llm, "log_url": LOG_URL}


@app.get("/run.jsonl")
def run_log():
    if os.path.exists(LOG_PATH):
        return FileResponse(LOG_PATH, media_type="text/plain; charset=utf-8", filename="run.jsonl")
    return PlainTextResponse("", media_type="text/plain")


@app.get("/")
def root():
    return {"service": "data-analyst-telegram-bot", "log_url": LOG_URL}
