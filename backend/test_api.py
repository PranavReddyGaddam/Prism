"""
API test script — run with: python test_api.py
Requires the backend server to be running on http://localhost:8000
"""
import requests
import json
import time

BASE = "http://localhost:8000"
TIMEOUT = 300  # seconds


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def ok(label: str, data=None):
    print(f"  ✅ {label}")
    if data:
        print(f"     {json.dumps(data, ensure_ascii=False)[:200]}")


def fail(label: str, detail=""):
    print(f"  ❌ {label}")
    if detail:
        print(f"     {str(detail)[:300]}")


# ── Health ────────────────────────────────────────────────────────────────────
section("Health")
try:
    r = requests.get(f"{BASE}/health", timeout=10)
    if r.status_code == 200:
        ok("GET /health", r.json())
    else:
        fail("GET /health", r.text)
except Exception as e:
    fail("GET /health", e)

# ── Models ────────────────────────────────────────────────────────────────────
section("Models")
try:
    r = requests.get(f"{BASE}/models/", timeout=10)
    if r.status_code == 200:
        models = r.json().get("models", [])
        ok(f"GET /models/ — {len(models)} models listed")
        for m in models:
            print(f"     • {m['id']} — {m['name']}")
    else:
        fail("GET /models/", r.text)
except Exception as e:
    fail("GET /models/", e)

try:
    r = requests.get(f"{BASE}/models/phi3", timeout=10)
    if r.status_code == 200:
        ok("GET /models/phi3", r.json())
    else:
        fail("GET /models/phi3", r.text)
except Exception as e:
    fail("GET /models/phi3", e)

# ── Generation — Phi-3 ────────────────────────────────────────────────────────
section("Generation — Phi-3")
PHI_PROMPT = "What is 2 + 2? Give a brief answer."
try:
    t0 = time.time()
    r = requests.post(
        f"{BASE}/generate/",
        json={"model_id": "phi3", "prompt": PHI_PROMPT, "max_new_tokens": 128},
        timeout=TIMEOUT,
    )
    elapsed = time.time() - t0
    if r.status_code == 200:
        d = r.json()
        ok(f"POST /generate/ phi3 ({elapsed:.1f}s)")
        print(f"     response     : {d.get('response','')[:150]}")
        print(f"     final_answer : {d.get('final_answer','')[:150]}")
        print(f"     token_count  : {d.get('token_count')}")
    else:
        fail("POST /generate/ phi3", r.text)
except Exception as e:
    fail("POST /generate/ phi3", e)

# ── Generation — DeepSeek ─────────────────────────────────────────────────────
section("Generation — DeepSeek")
DS_PROMPT = "What is the capital of France? Think step by step."
try:
    t0 = time.time()
    r = requests.post(
        f"{BASE}/generate/",
        json={"model_id": "deepseek", "prompt": DS_PROMPT, "max_new_tokens": 128},
        timeout=TIMEOUT,
    )
    elapsed = time.time() - t0
    if r.status_code == 200:
        d = r.json()
        ok(f"POST /generate/ deepseek ({elapsed:.1f}s)")
        print(f"     thinking     : {str(d.get('thinking',''))[:150]}")
        print(f"     final_answer : {d.get('final_answer','')[:150]}")
        print(f"     token_count  : {d.get('token_count')}")
    else:
        fail("POST /generate/ deepseek", r.text)
except Exception as e:
    fail("POST /generate/ deepseek", e)

# ── Explainability — Phi-3 ────────────────────────────────────────────────────
section("Explainability — Phi-3 (short prompt)")
EXPLAIN_BODY = {"model_id": "phi3", "prompt": "What is 2+2?", "max_new_tokens": 32}

try:
    t0 = time.time()
    r = requests.post(f"{BASE}/explain/confidence", json=EXPLAIN_BODY, timeout=TIMEOUT)
    elapsed = time.time() - t0
    if r.status_code == 200:
        d = r.json()
        n = len(d.get("token_confidence", []))
        ok(f"POST /explain/confidence ({elapsed:.1f}s) — {n} tokens")
    else:
        fail("POST /explain/confidence", r.text)
except Exception as e:
    fail("POST /explain/confidence", e)

try:
    t0 = time.time()
    body = {**EXPLAIN_BODY, "attn_layer": 0, "attn_head": 0}
    r = requests.post(f"{BASE}/explain/attention", json=body, timeout=TIMEOUT)
    elapsed = time.time() - t0
    if r.status_code == 200:
        d = r.json()
        ok(f"POST /explain/attention ({elapsed:.1f}s) — {len(d.get('tokens',[]))} tokens, matrix {len(d.get('matrix',[]))}x{len(d.get('matrix',[[]])[0]) if d.get('matrix') else 0}")
    else:
        fail("POST /explain/attention", r.text)
except Exception as e:
    fail("POST /explain/attention", e)

try:
    t0 = time.time()
    r = requests.post(f"{BASE}/explain/logit-lens", json=EXPLAIN_BODY, timeout=TIMEOUT)
    elapsed = time.time() - t0
    if r.status_code == 200:
        d = r.json()
        n = len(d.get("logit_lens", []))
        ok(f"POST /explain/logit-lens ({elapsed:.1f}s) — {n} layers")
    else:
        fail("POST /explain/logit-lens", r.text)
except Exception as e:
    fail("POST /explain/logit-lens", e)

try:
    t0 = time.time()
    r = requests.post(f"{BASE}/explain/hidden-states", json=EXPLAIN_BODY, timeout=TIMEOUT)
    elapsed = time.time() - t0
    if r.status_code == 200:
        d = r.json()
        n = len(d.get("hidden_state_norms", []))
        ok(f"POST /explain/hidden-states ({elapsed:.1f}s) — {n} layers")
    else:
        fail("POST /explain/hidden-states", r.text)
except Exception as e:
    fail("POST /explain/hidden-states", e)

try:
    t0 = time.time()
    r = requests.post(f"{BASE}/explain/attribution", json=EXPLAIN_BODY, timeout=TIMEOUT)
    elapsed = time.time() - t0
    if r.status_code == 200:
        d = r.json()
        n = len(d.get("gradient_attribution", []))
        ok(f"POST /explain/attribution ({elapsed:.1f}s) — {n} tokens")
    else:
        fail("POST /explain/attribution", r.text)
except Exception as e:
    fail("POST /explain/attribution", e)

# ── Error handling ────────────────────────────────────────────────────────────
section("Error Handling")
try:
    r = requests.post(
        f"{BASE}/generate/",
        json={"model_id": "nonexistent_model", "prompt": "test"},
        timeout=10,
    )
    if r.status_code == 400:
        ok("Invalid model_id returns 400", {"detail": r.json().get("detail","")[:80]})
    else:
        fail(f"Expected 400, got {r.status_code}", r.text)
except Exception as e:
    fail("Error handling test", e)

print(f"\n{'='*60}")
print("  Done")
print('='*60)
