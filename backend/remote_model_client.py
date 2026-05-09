import os
import httpx
from typing import Dict

MODEL_BASE_URL = os.getenv("MODEL_BASE_URL", "")
# Pod runs one inference at a time; four parallel UI calls queue — allow headroom.
TIMEOUT = float(os.getenv("MODEL_HTTP_TIMEOUT", "600"))

# RunPod proxy sits behind Cloudflare — these headers bypass the browser challenge page.
_HEADERS = {
    "Content-Type": "application/json",
    "User-Agent": "python-httpx/prism-backend",
    "ngrok-skip-browser-warning": "true",
    "bypass-tunnel-reminder": "true",
}

VALID_MODEL_IDS = {
    "qwen-2.5-math-7b",
    "deepseek-math-7b",
    "internlm2-math-7b",
    "wizardmath-7b",
}


def format_math_prompt(problem: str) -> str:
    return f"### Problem:\n{problem}\n### Solution:\n"


async def get_model_response(model_id: str, prompt: str, max_new_tokens: int = 256) -> Dict:
    """Call POST /generate on the RunPod bridge for any of the four math models."""
    if not MODEL_BASE_URL:
        raise ValueError("MODEL_BASE_URL environment variable not set")
    formatted = format_math_prompt(prompt)
    async with httpx.AsyncClient(timeout=TIMEOUT, headers=_HEADERS) as client:
        r = await client.post(
            f"{MODEL_BASE_URL}/generate",
            json={"model_id": model_id, "prompt": formatted, "max_new_tokens": max_new_tokens},
        )
        r.raise_for_status()
        return r.json()


async def check_remote_model_health() -> Dict:
    if not MODEL_BASE_URL:
        return {"status": "error", "message": "MODEL_BASE_URL not configured"}
    try:
        async with httpx.AsyncClient(timeout=10.0, headers=_HEADERS) as client:
            r = await client.get(f"{MODEL_BASE_URL}/health")
            r.raise_for_status()
            return r.json()
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ── Explainability (stubs on pod side for now) ────────────────────────────────

async def get_token_confidence(model_id: str, prompt: str, response: str) -> Dict:
    if not MODEL_BASE_URL:
        raise ValueError("MODEL_BASE_URL not set")
    async with httpx.AsyncClient(timeout=TIMEOUT, headers=_HEADERS) as client:
        r = await client.post(
            f"{MODEL_BASE_URL}/explain/confidence",
            json={"model_id": model_id, "prompt": prompt, "response": response},
        )
        r.raise_for_status()
        return r.json()


async def get_attention_weights(
    model_id: str, prompt: str, attn_layer: int = 0, attn_head: int = 0
) -> Dict:
    if not MODEL_BASE_URL:
        raise ValueError("MODEL_BASE_URL not set")
    async with httpx.AsyncClient(timeout=TIMEOUT, headers=_HEADERS) as client:
        r = await client.post(
            f"{MODEL_BASE_URL}/explain/attention",
            json={
                "model_id": model_id,
                "prompt": prompt,
                "attn_layer": attn_layer,
                "attn_head": attn_head,
            },
        )
        r.raise_for_status()
        return r.json()


async def get_logit_lens(model_id: str, prompt: str, response: str) -> Dict:
    if not MODEL_BASE_URL:
        raise ValueError("MODEL_BASE_URL not set")
    async with httpx.AsyncClient(timeout=TIMEOUT, headers=_HEADERS) as client:
        r = await client.post(
            f"{MODEL_BASE_URL}/explain/logit-lens",
            json={"model_id": model_id, "prompt": prompt, "response": response},
        )
        r.raise_for_status()
        return r.json()


async def get_hidden_states(model_id: str, prompt: str) -> Dict:
    if not MODEL_BASE_URL:
        raise ValueError("MODEL_BASE_URL not set")
    async with httpx.AsyncClient(timeout=TIMEOUT, headers=_HEADERS) as client:
        r = await client.post(
            f"{MODEL_BASE_URL}/explain/hidden-states",
            json={"model_id": model_id, "prompt": prompt},
        )
        r.raise_for_status()
        return r.json()


async def get_gradient_attribution(model_id: str, prompt: str, response: str) -> Dict:
    if not MODEL_BASE_URL:
        raise ValueError("MODEL_BASE_URL not set")
    async with httpx.AsyncClient(timeout=TIMEOUT, headers=_HEADERS) as client:
        r = await client.post(
            f"{MODEL_BASE_URL}/explain/attribution",
            json={"model_id": model_id, "prompt": prompt, "response": response},
        )
        r.raise_for_status()
        return r.json()


# ── Post-hoc analysis (LIME / TokenSHAP / Counterfactual) ────────────────────

async def get_posthoc_lime(model_id: str, prompt: str, response: str = "", n_samples: int = 60) -> Dict:
    if not MODEL_BASE_URL:
        raise ValueError("MODEL_BASE_URL not set")
    async with httpx.AsyncClient(timeout=TIMEOUT, headers=_HEADERS) as client:
        r = await client.post(
            f"{MODEL_BASE_URL}/posthoc/lime",
            json={"model_id": model_id, "prompt": prompt, "response": response, "n_samples": n_samples},
        )
        r.raise_for_status()
        return r.json()


async def get_posthoc_tokenshap(model_id: str, prompt: str, response: str = "", n_samples: int = 50) -> Dict:
    if not MODEL_BASE_URL:
        raise ValueError("MODEL_BASE_URL not set")
    async with httpx.AsyncClient(timeout=TIMEOUT, headers=_HEADERS) as client:
        r = await client.post(
            f"{MODEL_BASE_URL}/posthoc/tokenshap",
            json={"model_id": model_id, "prompt": prompt, "response": response, "n_samples": n_samples},
        )
        r.raise_for_status()
        return r.json()


async def get_posthoc_counterfactual(model_id: str, prompt: str, response: str = "") -> Dict:
    if not MODEL_BASE_URL:
        raise ValueError("MODEL_BASE_URL not set")
    async with httpx.AsyncClient(timeout=TIMEOUT, headers=_HEADERS) as client:
        r = await client.post(
            f"{MODEL_BASE_URL}/posthoc/counterfactual",
            json={"model_id": model_id, "prompt": prompt, "response": response},
        )
        r.raise_for_status()
        return r.json()
