# Data-Analyst Telegram Bot

An LLM agent that answers data-analysis questions sent to it on Telegram.
Built for IIT Madras Tools in Data Science, Project 1 (Task 2).

## What it does

Message the bot a data-analysis question — data embedded inline, or a pointer to
a public dataset such as MOSPI. The agent works out the answer, fetching data and
running pandas/numpy code in a `run_python` tool when needed, and replies with
**exactly one JSON object and nothing else**:

```json
{"answer": {"state": "Assam"}, "log_url": "https://<host>/run.jsonl"}
```

- `answer` — shaped exactly as the message asks.
- `log_url` — a public, `wget`-able JSONL log of every agent step (question, tool
  calls, tool output, final answer), one JSON object per line.

Multi-turn exchanges are supported: per-chat history is kept, every message gets
a reply, and the agent answers the latest message in context.

## Reply shaping

The grading harness compares the **whole** parsed reply object to the expected
answer, so the reply has to mirror the shape the message asked for:

| Message asks for | Bot replies |
|---|---|
| `{"answer": {...}, "log_url": "<url>"}` | `{"answer": {...}, "log_url": "<real url>"}` |
| `{"state": "<state name>"}` (no `log_url`) | `{"state": "Assam"}` |

`shape_reply()` handles both, and repairs the two common model mistakes —
forgetting the envelope, or wrapping an answer that was meant to be bare.
Set `FORCE_WRAP=1` to always emit the two-key envelope regardless.

## Architecture

`bot.py` is the whole service:

- FastAPI app serving `/health` and `/run.jsonl` (the public agent log)
- background thread long-polling the Telegram Bot API (`getUpdates`; the webhook
  is deleted at startup so polling always works)
- agentic loop over an OpenAI-compatible chat API with a `run_python` tool
  (pandas, numpy, scipy, requests, BeautifulSoup, lxml, openpyxl; network on)
- wall-clock budget (`ANSWER_BUDGET`, default 150 s) that forces a final answer
  before the harness's per-exchange timeout expires
- keep-warm self-ping so a free host never idles out

## Configuration

| Variable | Purpose |
|---|---|
| `BOT_TOKEN` | from @BotFather |
| `AIPIPE_TOKEN` | OpenAI-compatible API key |
| `MODEL` | default `gpt-4o-mini` |
| `MODEL_BASE_URL` | default `https://aipipe.org/openai/v1` |
| `BASE_URL` | public URL of this service — `log_url` is derived from it |
| `LOG_PATH` | default `/tmp/run.jsonl` |
| `ANSWER_BUDGET`, `MAX_AGENT_STEPS`, `PY_TIMEOUT`, `FORCE_WRAP` | tuning |

## Run locally

```bash
pip install -r requirements.txt
cp .env.example .env        # fill it in, then export the vars
uvicorn bot:app --host 0.0.0.0 --port 8000 --workers 1
```

Run **one worker only** — more than one process means more than one poller
competing for the same updates.

Offline check (no Telegram needed):

```bash
AIPIPE_TOKEN=... BASE_URL=https://your-host python3 smoke_test.py
```

## Deploy

Any always-on host works. `render.yaml` (Render) and `Dockerfile` (Hugging Face
Spaces, Fly.io, Railway, any VPS) are both included. Set `BASE_URL` to the
service's own public URL after the first deploy.

## Test against the real grader

```bash
git clone https://github.com/Jivraj-18/tds-p1-t2-2026-telegram-bot
cd tds-p1-t2-2026-telegram-bot
pip install -r requirements.txt
cp ../evals/questions.json evals/questions.json   # replace the placeholder log_urls first
printf 'email,github_url,telegram_bot_username\nme@example.com,https://github.com/you/repo,your_bot\n' > students.csv
python3 generate.py --students students.csv
python3 collect.py  --students students.csv
python3 grade.py    --students students.csv
```

Needs `TELEGRAM_API_ID` / `TELEGRAM_API_HASH` from my.telegram.org and a session
string from `python3 login.py` — the grader logs in as a real user account,
because bots can't message bots.
