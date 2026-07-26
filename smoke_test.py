#!/usr/bin/env python3
"""Offline smoke test - runs the agent loop directly, no Telegram needed.

    BOT_TOKEN=x AIPIPE_TOKEN=... BASE_URL=https://your-host python3 smoke_test.py

Prints the exact string the bot would send back. It must be one JSON object.
"""
import json
import os
import sys

os.environ.setdefault("LOG_PATH", "test_run.jsonl")
import bot  # noqa: E402

QUESTIONS = [
    'Which state has the highest maternal mortality rate based on MOSPI data? '
    'Reply with ONLY this JSON object and nothing else: '
    '{"answer": {"state": "<state name>"}, "log_url": "<public wget-able URL to your agent\'s JSONL log>"}',

    'The values are [12, 47, 3, 88, 21]. Give the mean rounded to 2 decimals and the median. '
    'Reply with ONLY this JSON object and nothing else: '
    '{"answer": {"mean": <number>, "median": <number>}, "log_url": "<url>"}',

    'Reply with ONLY a JSON object like {"state": "<state name>"}: which state has the '
    'highest maternal mortality rate per MOSPI/SRS?',
]

for i, q in enumerate(sys.argv[1:] or QUESTIONS):
    print(f"\n--- Q{i} ---\n{q}\n")
    reply = bot.solve(1000 + i, q)
    print("REPLY:", reply)
    try:
        json.loads(reply)
        print("OK: parses as exactly one JSON object")
    except json.JSONDecodeError as e:
        print("FAIL: not valid JSON —", e)
