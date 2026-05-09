#!/usr/bin/env python3
"""
HF bridge for four math LLMs on RunPod.

Run on the pod, bind 0.0.0.0, expose the HTTP port in RunPod UI.

Environment — set the path for each model (HF snapshot dir on pod):
  RUNPOD_PATH_QWEN        /workspace/models/qwen25-math-7b
  RUNPOD_PATH_DEEPSEEK    /workspace/models/deepseek-math-7b
  RUNPOD_PATH_INTERNLM2   /workspace/models/internlm2-math-plus-7b
  RUNPOD_PATH_WIZARDMATH  /workspace/models/wizardmath-7b-v1.1

  RUNPOD_TRUST_QWEN=0|1       (set 1 if the checkpoint needs trust_remote_code)
  RUNPOD_TRUST_DEEPSEEK=0|1
  RUNPOD_TRUST_INTERNLM2=0|1
  RUNPOD_TRUST_WIZARDMATH=0|1

Backend on your Mac:
  MODEL_BASE_URL=https://<id>-8000.proxy.runpod.net   (no trailing slash)

Start:
  cd /workspace/Prism/scripts
  export RUNPOD_PATH_QWEN=/workspace/models/qwen25-math-7b
  export RUNPOD_PATH_DEEPSEEK=/workspace/models/deepseek-math-7b
  export RUNPOD_PATH_INTERNLM2=/workspace/models/internlm2-math-plus-7b
  export RUNPOD_PATH_WIZARDMATH=/workspace/models/wizardmath-7b-v1.1
  uvicorn runpod_hf_server:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os
import re
import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="RunPod HF bridge — math models", version="0.2.0")

# Keep all loaded models in memory — 4x7B bfloat16 ~56GB fits in 80GB VRAM
_cache: Dict[str, Tuple[Any, Any]] = {}

# One inference at a time — concurrent GPU generate/load races CUDA and often returns 500
# when the frontend fires four parallel POST /generate calls.
_GEN_LOCK = threading.Lock()

# Maps frontend model_id → (env var for path, env var for trust_remote_code)
MODEL_ENV: Dict[str, Tuple[str, str]] = {
    "qwen-2.5-math-7b":  ("RUNPOD_PATH_QWEN",      "RUNPOD_TRUST_QWEN"),
    "deepseek-math-7b":  ("RUNPOD_PATH_DEEPSEEK",   "RUNPOD_TRUST_DEEPSEEK"),
    "internlm2-math-7b": ("RUNPOD_PATH_INTERNLM2",  "RUNPOD_TRUST_INTERNLM2"),
    "wizardmath-7b":     ("RUNPOD_PATH_WIZARDMATH",  "RUNPOD_TRUST_WIZARDMATH"),
}


def _trust(env_name: str) -> bool:
    return os.getenv(env_name, "0").strip() in ("1", "true", "True", "yes")


def _dtype():
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def _load(path: str, trust: bool) -> Tuple[Any, Any]:
    if path in _cache:
        return _cache[path]
    if not path or not os.path.isdir(path):
        raise HTTPException(status_code=500, detail=f"Invalid model path: {path!r}")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=trust)
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=_dtype(),
        device_map="auto",
        trust_remote_code=trust,
    )
    model.eval()
    _cache[path] = (tok, model)
    return _cache[path]


class GenerateBody(BaseModel):
    model_id: str
    prompt: str
    max_new_tokens: Optional[int] = 512


class ExplainBody(BaseModel):
    model_id: str
    prompt: str
    response: Optional[str] = None



@app.get("/health")
def health():
    configured = [mid for mid, (penv, _) in MODEL_ENV.items() if os.getenv(penv, "").strip()]
    loaded = [mid for mid, (penv, _) in MODEL_ENV.items() if os.getenv(penv, "").strip() and os.getenv(penv, "").strip() in _cache]
    return {"status": "ok", "models_configured": configured, "models_loaded": loaded}


@app.post("/generate")
def generate(body: GenerateBody):
    if body.model_id not in MODEL_ENV:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model_id {body.model_id!r}. Valid: {list(MODEL_ENV.keys())}",
        )
    path_env, trust_env = MODEL_ENV[body.model_id]
    path = os.getenv(path_env, "").strip()
    if not path:
        raise HTTPException(status_code=500, detail=f"{path_env} is not set on the pod")

    with _GEN_LOCK:
        tok, model = _load(path, _trust(trust_env))
        max_new = min(int(body.max_new_tokens or 2000), 2000)
        pad = tok.pad_token_id or tok.eos_token_id
        inputs = tok(body.prompt, return_tensors="pt")
        inputs = {k: v.to(device=model.device, dtype=model.dtype) if v.is_floating_point() else v.to(model.device) for k, v in inputs.items()}
        try:
            with torch.inference_mode():
                out = model.generate(
                    **inputs,
                    max_new_tokens=max_new,
                    min_new_tokens=64,
                    do_sample=False,
                    pad_token_id=pad,
                )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
            raise HTTPException(status_code=503, detail=f"GPU inference failed: {e}") from e
        # Decode only the newly generated tokens (slice off the prompt input tokens)
        n_input = inputs["input_ids"].shape[-1]
        generated_ids = out[0][n_input:]
        text = _truncate_response(tok.decode(generated_ids, skip_special_tokens=True))
        torch.cuda.empty_cache()
    return {"model_id": body.model_id, "response": text}


def _resolve(body: ExplainBody):
    if body.model_id not in MODEL_ENV:
        raise HTTPException(status_code=400, detail=f"Unknown model_id {body.model_id!r}")
    path_env, trust_env = MODEL_ENV[body.model_id]
    path = os.getenv(path_env, "").strip()
    if not path:
        raise HTTPException(status_code=500, detail=f"{path_env} not set")
    return _load(path, _trust(trust_env))


def _encode(tok, model, text: str):
    enc = tok(text, return_tensors="pt")
    return {k: v.to(device=model.device, dtype=model.dtype) if v.is_floating_point() else v.to(model.device)
            for k, v in enc.items()}


@app.post("/explain/confidence")
def ex_confidence(body: ExplainBody):
    tok, model = _resolve(body)
    full = (body.prompt + " " + body.response) if body.response else body.prompt
    with _GEN_LOCK:
        inputs = _encode(tok, model, full)
        with torch.inference_mode():
            out = model(**inputs)
        logits = out.logits[0]  # (seq, vocab)
        probs = torch.softmax(logits, dim=-1)
        ids = inputs["input_ids"][0]
        token_confidence = []
        for i, tok_id in enumerate(ids[1:], start=1):
            conf = probs[i - 1, tok_id].item()
            token_confidence.append({"token": tok.decode([tok_id.item()]), "confidence": conf})
        torch.cuda.empty_cache()
    return {"token_confidence": token_confidence}


@app.post("/explain/hidden-states")
def ex_hidden(body: ExplainBody):
    tok, model = _resolve(body)
    with _GEN_LOCK:
        inputs = _encode(tok, model, body.prompt)
        input_ids = inputs["input_ids"][0]
        tokens = [tok.decode([tid], skip_special_tokens=False) for tid in input_ids.tolist()]

        with torch.inference_mode():
            out = model(**inputs, output_hidden_states=True)

        all_hs = out.hidden_states  # tuple of (1, seq_len, hidden_dim)
        n_layers = len(all_hs)
        seq_len = all_hs[0].shape[1]

        hidden_state_norms = []
        prev_mean = None
        for i, hs in enumerate(all_hs):
            norms_per_token = hs[0].float().norm(dim=-1).tolist()  # (seq_len,)
            mean_norm = sum(norms_per_token) / len(norms_per_token)
            delta = abs(mean_norm - prev_mean) if prev_mean is not None else 0.0
            prev_mean = mean_norm
            hidden_state_norms.append({
                "layer": i,
                "norm": mean_norm,
                "delta": delta,
                "token_norms": [round(v, 3) for v in norms_per_token],
            })

        torch.cuda.empty_cache()
    return {
        "hidden_state_norms": hidden_state_norms,
        "tokens": tokens,
        "n_layers": n_layers,
        "seq_len": seq_len,
    }


def _get_lm_head(model):
    """Get the LM head regardless of architecture naming."""
    for name in ("lm_head", "output", "embed_out", "head"):
        if hasattr(model, name):
            return getattr(model, name)
    raise AttributeError(f"Cannot find lm_head on {type(model).__name__}")


@app.post("/explain/logit-lens")
def ex_logit(body: ExplainBody):
    tok, model = _resolve(body)
    with _GEN_LOCK:
        inputs = _encode(tok, model, body.prompt)
        with torch.inference_mode():
            out = model(**inputs, output_hidden_states=True)
        lm_head = _get_lm_head(model)
        logit_lens = []
        for layer_i, hs in enumerate(out.hidden_states):
            # Clone to detach from inference_mode graph before passing through lm_head
            hs_clone = hs[0].clone().to(lm_head.weight.dtype)
            with torch.no_grad():
                layer_logits = lm_head(hs_clone)
            layer_probs = torch.softmax(layer_logits, dim=-1)
            for pos in range(min(5, hs.shape[1])):
                top_id = layer_probs[pos].argmax().item()
                top_prob = layer_probs[pos, top_id].item()
                logit_lens.append({
                    "layer": layer_i,
                    "word_position": pos,
                    "predicted_token": tok.decode([top_id]),
                    "probability": top_prob,
                })
        torch.cuda.empty_cache()
    return {"logit_lens": logit_lens}


@app.post("/explain/attribution")
def ex_attr(body: ExplainBody):
    """Attention-rollout attribution with per-position top/bottom token predictions."""
    tok, model = _resolve(body)
    full = (body.prompt + " " + body.response) if body.response else body.prompt
    TOP_K = 5  # top and bottom predictions per position

    with _GEN_LOCK:
        inputs = _encode(tok, model, full)
        with torch.inference_mode():
            out = model(**inputs, output_attentions=True, output_hidden_states=True)
        torch.cuda.empty_cache()

        tokens = [tok.decode([i]) for i in inputs["input_ids"][0].tolist()]
        seq_len = len(tokens)

        # Attention rollout scores
        attn_stack = torch.stack([a[0].float().mean(dim=0) for a in out.attentions])
        rollout = torch.eye(attn_stack.shape[-1], device=attn_stack.device)
        for a in attn_stack:
            a = a + torch.eye(a.shape[-1], device=a.device)
            a = a / a.sum(dim=-1, keepdim=True)
            rollout = a @ rollout
        scores = rollout[-1]
        scores = (scores / scores.max()).tolist()
        gradient_attribution = [{"token": t, "score": s} for t, s in zip(tokens, scores)]

        # Per-position top/bottom token predictions at final layer
        lm_head = _get_lm_head(model)
        last_hidden = out.hidden_states[-1][0].clone()  # (seq, hidden)
        with torch.no_grad():
            logits = lm_head(last_hidden.to(lm_head.weight.dtype))  # (seq, vocab)
        probs = torch.softmax(logits, dim=-1)

        position_predictions = []
        for pos in range(seq_len):
            pos_probs = probs[pos]
            top_ids = pos_probs.topk(TOP_K).indices.tolist()
            top_probs = pos_probs.topk(TOP_K).values.tolist()
            # Bottom k = lowest prob non-zero tokens
            bottom_ids = pos_probs.topk(TOP_K, largest=False).indices.tolist()
            bottom_probs = pos_probs.topk(TOP_K, largest=False).values.tolist()
            position_predictions.append({
                "position": pos,
                "token": tokens[pos],
                "top": [{"token": tok.decode([tid]), "probability": p, "rank": r + 1}
                        for r, (tid, p) in enumerate(zip(top_ids, top_probs))],
                "bottom": [{"token": tok.decode([tid]), "probability": p, "rank": r + 1}
                           for r, (tid, p) in enumerate(zip(bottom_ids, bottom_probs))],
            })

        # Per-layer attention weights per token (for input/output feature weights)
        num_layers = len(out.attentions)
        layer_attention = []
        for layer_i, attn in enumerate(out.attentions):
            # Mean over heads, take last token's attention to each input
            mean_attn = attn[0].float().mean(dim=0)  # (seq, seq)
            layer_attention.append({
                "layer": layer_i,
                "weights": [
                    {"token": tokens[src], "weight": float(mean_attn[-1, src])}
                    for src in range(seq_len)
                ]
            })

        torch.cuda.empty_cache()

    return {
        "gradient_attribution": gradient_attribution,
        "position_predictions": position_predictions,
        "layer_attention": layer_attention,
        "tokens": tokens,
    }


@app.post("/explain/attention")
def ex_attn(body: ExplainBody):
    tok, model = _resolve(body)
    layer = 0
    head = 0
    with _GEN_LOCK:
        inputs = _encode(tok, model, body.prompt)
        with torch.inference_mode():
            out = model(**inputs, output_attentions=True)
        attn = out.attentions[layer][0, head].float().tolist()
        tokens = [tok.decode([i]) for i in inputs["input_ids"][0].tolist()]
    return {"tokens": tokens, "matrix": attn, "layer": layer, "head": head}


# ─────────────────────────────────────────────────────────────────────────────
# Post-hoc analysis endpoints (LIME, TokenSHAP, counterfactuals)
# All forward-pass-only — no gradients, no new model loads.
# ─────────────────────────────────────────────────────────────────────────────


class PostHocBody(BaseModel):
    model_id: str
    prompt: str
    response: Optional[str] = None
    n_samples: Optional[int] = None
    max_new_tokens: Optional[int] = 96


_FINAL_NUMBER_RE = re.compile(r"\\boxed\s*\{\s*([+-]?\d+(?:\.\d+)?)", re.IGNORECASE)
_ANY_NUMBER_RE = re.compile(r"[+-]?\d+(?:\.\d+)?")
# Matches "answer is 12", "= 12", "therefore 12", etc. near the end of a response
_ANSWER_PHRASE_RE = re.compile(
    r"(?:answer(?:\s+is)?|therefore|thus|so|=)\s*([+-]?\d+(?:\.\d+)?)\s*[.\n]?\s*$",
    re.IGNORECASE,
)


def _extract_final_answer(text: str) -> Optional[str]:
    if not text:
        return None
    # 1. Prefer \boxed{} — most reliable final answer marker
    m = _FINAL_NUMBER_RE.search(text)
    if m:
        return m.group(1)
    # 2. Answer phrase at the end of text
    m = _ANSWER_PHRASE_RE.search(text)
    if m:
        return m.group(1)
    # 3. Last number in the final 80 characters only — avoids mid-reasoning leakage
    tail = text[-80:]
    nums = _ANY_NUMBER_RE.findall(tail)
    return nums[-1] if nums else None


def _short_generate(model, tok, prompt: str, max_new: int = 2000) -> str:
    pad = tok.pad_token_id or tok.eos_token_id
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {
        k: v.to(device=model.device, dtype=model.dtype) if v.is_floating_point() else v.to(model.device)
        for k, v in inputs.items()
    }
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new,
            min_new_tokens=64,
            do_sample=False,
            pad_token_id=pad,
        )
    full = tok.decode(out[0], skip_special_tokens=True)
    # Strip echoed prompt, then truncate any leaked next-problem text
    text = full[len(prompt):].strip() if full.startswith(prompt) else full.strip()
    return _truncate_response(text)


_PROMPT_PREFIX = "### Problem:\n"
_PROMPT_SUFFIX = "\n### Solution:\n"
# Markers that indicate the model has started generating a new problem — truncate here
_STOP_PATTERNS = ["### Problem:", "### Solution:", "\n##"]

def _truncate_response(text: str) -> str:
    """Cut off any text where the model starts generating a new problem/prompt."""
    for pat in _STOP_PATTERNS:
        idx = text.find(pat)
        if idx != -1:
            text = text[:idx]
    return text.strip()

def _unwrap_prompt(prompt: str) -> str:
    """Strip ### Problem:/Solution: wrapper if present, returning raw problem text."""
    p = prompt.strip()
    if p.startswith(_PROMPT_PREFIX):
        p = p[len(_PROMPT_PREFIX):]
    if _PROMPT_SUFFIX.strip() in p:
        p = p[:p.rfind(_PROMPT_SUFFIX.strip())]
    return p.strip()

def _wrap_prompt(raw: str) -> str:
    return f"{_PROMPT_PREFIX}{raw}{_PROMPT_SUFFIX}"

def _resolve_model(model_id: str):
    if model_id not in MODEL_ENV:
        raise HTTPException(status_code=400, detail=f"Unknown model_id {model_id!r}")
    path_env, trust_env = MODEL_ENV[model_id]
    path = os.getenv(path_env, "").strip()
    if not path:
        raise HTTPException(status_code=500, detail=f"{path_env} not set")
    return _load(path, _trust(trust_env))


def _weighted_ridge(X: np.ndarray, y: np.ndarray, w: np.ndarray, alpha: float = 1.0):
    """
    Closed-form weighted ridge regression on a design matrix that includes
    an intercept column. Returns (coef, intercept, r_squared) — no sklearn.

    Solves: argmin_β  Σ wᵢ(yᵢ - Xβ)²  +  α‖β‖²   (intercept un-penalized)
    """
    n, d = X.shape
    X_aug = np.hstack([np.ones((n, 1)), X])           # prepend intercept column
    W = np.diag(w)
    XtWX = X_aug.T @ W @ X_aug
    # Penalty: skip the intercept (index 0)
    pen = alpha * np.eye(d + 1)
    pen[0, 0] = 0.0
    try:
        beta = np.linalg.solve(XtWX + pen, X_aug.T @ W @ y)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(XtWX + pen, X_aug.T @ W @ y, rcond=None)[0]
    intercept = float(beta[0])
    coef = beta[1:].astype(float)

    # Weighted R²
    y_pred = X_aug @ beta
    y_mean = float((w * y).sum() / max(w.sum(), 1e-9))
    ss_res = float((w * (y - y_pred) ** 2).sum())
    ss_tot = float((w * (y - y_mean) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    return coef, intercept, r2


@app.post("/posthoc/lime")
def posthoc_lime(body: PostHocBody):
    """
    Word-level LIME via random masking + closed-form weighted Ridge regression.
    Target: match (1.0/0.0) between perturbed answer and reference answer.
    """
    tok, model = _resolve_model(body.model_id)
    n_samples = max(6, min(int(body.n_samples or 16), 24))
    max_new = min(int(body.max_new_tokens or 2000), 2000)

    raw_problem = _unwrap_prompt(body.prompt)
    words = raw_problem.split()
    n_words = len(words)
    if n_words < 2:
        raise HTTPException(status_code=400, detail="Prompt too short for LIME")

    with _GEN_LOCK:
        # Reference answer (unperturbed)
        ref_response = body.response or _short_generate(model, tok, _wrap_prompt(raw_problem), max_new=max_new)
        ref_answer = _extract_final_answer(ref_response)

        rng = np.random.default_rng(42)
        masks = rng.integers(0, 2, size=(n_samples, n_words)).astype(np.float64)
        masks[0] = np.ones(n_words, dtype=np.float64)  # sample 0 = full prompt

        try:
            ref_val = float(ref_answer) if ref_answer is not None else None
        except ValueError:
            ref_val = None

        targets = np.zeros(n_samples, dtype=np.float64)
        for i in range(n_samples):
            kept = [w for w, k in zip(words, masks[i]) if k == 1]
            if not kept:
                targets[i] = 0.0
                continue
            perturbed_prompt = _wrap_prompt(" ".join(kept))
            try:
                resp = _short_generate(model, tok, perturbed_prompt, max_new=max_new)
                ans = _extract_final_answer(resp)
                if ref_val is not None and ans is not None:
                    try:
                        # Continuous target: 1 - |perturbed - ref| / (|ref| + 1)
                        # Full prompt → target≈1, wrong answer → target<1, no answer → 0
                        targets[i] = max(0.0, 1.0 - abs(float(ans) - ref_val) / (abs(ref_val) + 1.0))
                    except ValueError:
                        targets[i] = 1.0 if ans == ref_answer else 0.0
                else:
                    targets[i] = 1.0 if (ans is not None and ans == ref_answer) else 0.0
            except Exception:
                targets[i] = 0.0
            if i % 16 == 0:
                torch.cuda.empty_cache()

        # Cosine-distance kernel weighting against the all-ones reference mask
        ref_mask = np.ones(n_words)
        ref_norm = np.linalg.norm(ref_mask) + 1e-9
        dists = np.empty(n_samples)
        for i, m in enumerate(masks):
            mn = np.linalg.norm(m)
            dists[i] = 1.0 - (np.dot(m, ref_mask) / (mn * ref_norm)) if mn > 0 else 1.0
        kernel_width = 0.75
        weights = np.exp(-(dists ** 2) / (kernel_width ** 2))

        try:
            coef, _intercept, r_squared = _weighted_ridge(masks, targets, weights, alpha=1.0)
            attributions = coef.tolist()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Ridge fit failed: {e}") from e

        torch.cuda.empty_cache()

    return {
        "method": "lime",
        "words": words,
        "attributions": attributions,
        "reference_answer": ref_answer,
        "reference_response": ref_response,
        "n_samples": n_samples,
        "r_squared": float(r_squared),
    }


@app.post("/posthoc/tokenshap")
def posthoc_tokenshap(body: PostHocBody):
    """
    Word-level Shapley values via Monte Carlo coalition sampling.
    Estimates each word's marginal contribution to keeping the answer correct.
    """
    tok, model = _resolve_model(body.model_id)
    n_samples = max(6, min(int(body.n_samples or 16), 24))
    max_new = min(int(body.max_new_tokens or 2000), 2000)

    raw_problem = _unwrap_prompt(body.prompt)
    words = raw_problem.split()
    n_words = len(words)
    if n_words < 2:
        raise HTTPException(status_code=400, detail="Prompt too short for TokenSHAP")

    with _GEN_LOCK:
        ref_response = body.response or _short_generate(model, tok, _wrap_prompt(raw_problem), max_new=max_new)
        ref_answer = _extract_final_answer(ref_response)
        try:
            ref_val = float(ref_answer) if ref_answer is not None else None
        except ValueError:
            ref_val = None

        rng = np.random.default_rng(7)
        marginal_sums = np.zeros(n_words, dtype=np.float64)
        marginal_counts = np.zeros(n_words, dtype=np.int64)

        coalition_cache: Dict[Tuple[int, ...], float] = {(): 0.0}

        def value_of(coal_idx: Tuple[int, ...]) -> float:
            if coal_idx in coalition_cache:
                return coalition_cache[coal_idx]
            if not coal_idx:
                return 0.0
            kept = [words[j] for j in coal_idx]
            perturbed = _wrap_prompt(" ".join(kept))
            try:
                resp = _short_generate(model, tok, perturbed, max_new=max_new)
                ans = _extract_final_answer(resp)
                if ref_val is not None and ans is not None:
                    try:
                        v = max(0.0, 1.0 - abs(float(ans) - ref_val) / (abs(ref_val) + 1.0))
                    except ValueError:
                        v = 1.0 if ans == ref_answer else 0.0
                else:
                    v = 1.0 if (ans is not None and ans == ref_answer) else 0.0
            except Exception:
                v = 0.0
            coalition_cache[coal_idx] = v
            return v

        # n_samples permutations × n_words marginals = many calls; cap by truncating early
        max_calls = n_samples
        calls_made = 0

        for perm_i in range(n_samples):
            perm = rng.permutation(n_words)
            current: List[int] = []
            prev_v = 0.0
            for idx in perm:
                if calls_made >= max_calls:
                    break
                current.append(int(idx))
                key = tuple(sorted(current))
                v = value_of(key)
                calls_made += 1
                marginal_sums[idx] += (v - prev_v)
                marginal_counts[idx] += 1
                prev_v = v
            if calls_made >= max_calls:
                break
            if perm_i % 4 == 0:
                torch.cuda.empty_cache()

        with np.errstate(invalid="ignore", divide="ignore"):
            shapley = np.where(marginal_counts > 0, marginal_sums / marginal_counts, 0.0)

        torch.cuda.empty_cache()

    return {
        "method": "tokenshap",
        "words": words,
        "attributions": shapley.astype(float).tolist(),
        "reference_answer": ref_answer,
        "reference_response": ref_response,
        "n_samples": int(calls_made),
    }


MAX_COUNTERFACTUAL_EDITS = 3


@app.post("/posthoc/counterfactual")
def posthoc_counterfactual(body: PostHocBody):
    """
    Numeric-perturbation counterfactuals.
    Tries up to MAX_COUNTERFACTUAL_EDITS edits per request, one candidate per
    numeric span (largest numbers first — they tend to be the most semantically
    loaded). This keeps total forward passes per model bounded and predictable.
    """
    tok, model = _resolve_model(body.model_id)
    max_new = min(int(body.max_new_tokens or 2000), 2000)

    spans = [(m.start(), m.end(), m.group(0)) for m in _ANY_NUMBER_RE.finditer(body.prompt)]
    if not spans:
        return {
            "method": "counterfactual",
            "reference_answer": None,
            "reference_response": "",
            "edits": [],
            "note": "No numeric tokens in prompt — counterfactual generation skipped.",
        }

    def best_candidate(value: str) -> Optional[str]:
        """Pick a single, sensible perturbation for one numeric token."""
        try:
            if "." in value:
                f = float(value)
                return f"{f + 1:g}" if f != -1 else f"{f + 2:g}"
            n = int(value)
            # Prefer +1 / -1 / *2 in that order; keep edits minimal & interpretable
            for cand in (n + 1, n - 1, n * 2):
                if cand != n:
                    return str(cand)
        except ValueError:
            return None
        return None

    # Sort spans by the absolute magnitude of the number (largest first), then
    # take the first MAX_COUNTERFACTUAL_EDITS spans — one perturbation each.
    def magnitude(s):
        try:
            return abs(float(s[2]))
        except ValueError:
            return 0.0

    selected_spans = sorted(spans, key=magnitude, reverse=True)[:MAX_COUNTERFACTUAL_EDITS]

    with _GEN_LOCK:
        ref_response = body.response or _short_generate(model, tok, body.prompt, max_new=max_new)
        ref_answer = _extract_final_answer(ref_response)

        edits: List[Dict[str, Any]] = []
        for span_start, span_end, original in selected_spans:
            cand = best_candidate(original)
            if cand is None:
                continue
            new_prompt = body.prompt[:span_start] + cand + body.prompt[span_end:]
            try:
                new_response = _short_generate(model, tok, new_prompt, max_new=max_new)
                new_answer = _extract_final_answer(new_response)
            except Exception as e:
                new_response = f"[generation failed: {e}]"
                new_answer = None
            flipped = (
                ref_answer is not None
                and new_answer is not None
                and new_answer != ref_answer
            )
            edits.append({
                "original_token": original,
                "replacement": cand,
                "position": span_start,
                "new_response": new_response,
                "new_answer": new_answer,
                "flipped": flipped,
            })
            torch.cuda.empty_cache()

        torch.cuda.empty_cache()

    edits.sort(key=lambda e: not e["flipped"])  # flipped edits first

    return {
        "method": "counterfactual",
        "reference_answer": ref_answer,
        "reference_response": ref_response,
        "edits": edits,
        "n_edits": len(edits),
        "n_flipped": sum(1 for e in edits if e["flipped"]),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
