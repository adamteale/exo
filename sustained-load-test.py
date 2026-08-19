#!/usr/bin/env python3
"""Sustained-load test: fire 5K-prompt calls at the 27B sharded in a loop.
Goal: reproduce the ring socket EFAULT + verify the crash-on-ring-abort fix
(the supervisor kills the runner → exo re-places → the 27B recovers).
"""
import json, time, urllib.request, sys

API = "http://127.0.0.1:52415/v1/chat/completions"
MODEL = "mlx-community/Qwen3.8-27B-OptiQ-4bit"
# ~5K-token prompt (reuses the verified working size)
PROMPT = ("You are analyzing a document. " + " ".join(
    f"Section {i}: The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs. How vexingly quick daft zebras jump!"
    for i in range(100)) + "\n\nWhat animal appears in the text above? Answer in one word.")

log = open("/tmp/exo-sustained-load.log", "w", buffering=1)
def logln(s):
    print(s, flush=True)
    log.write(s + "\n")

logln(f"starting sustained-load test: model={MODEL} prompt_chars={len(PROMPT)}")
logln(f"call | time_s | result | tokens")

consecutive_ok = 0
failures = 0
for i in range(1, 121):  # up to 120 calls (~60 min at 30s each)
    t0 = time.time()
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": PROMPT}],
                       "max_tokens": 20, "enable_thinking": False}).encode()
    req = urllib.request.Request(API, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            d = json.loads(r.read())
        c = d.get("choices", [{}])[0].get("message", {}).get("content", "?")
        u = d.get("usage", {})
        dt = time.time() - t0
        logln(f"{i:3d} | {dt:6.1f}s | OK | prompt={u.get('prompt_tokens')} completion={u.get('completion_tokens')} | {c!r:.40}")
        consecutive_ok += 1
    except Exception as e:
        dt = time.time() - t0
        failures += 1
        consecutive_ok = 0
        logln(f"{i:3d} | {dt:6.1f}s | FAIL | {type(e).__name__}: {str(e)[:120]}")
        # after a failure, wait a bit (the crash-on-ring-abort should restart the runner + exo re-places)
        time.sleep(30)
        continue

logln(f"done: {i} calls, {failures} failures, max consecutive OK={consecutive_ok}")