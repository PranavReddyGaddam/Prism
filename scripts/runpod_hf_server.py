#!/usr/bin/env python3
"""
Minimal HTTP bridge for the existing FastAPI backend (remote_model_client.py).

Run on RunPod (GPU), bind 0.0.0.0, expose the port in RunPod UI.

Environment (paths = local HF snapshot dirs):
  RUNPOD_PATH_BASE       — model for POST /base   (e.g. Qwen2.5-Math-7B)
  RUNPOD_PATH_FINETUNED — model for POST /finetuned (e.g. DeepSeek-Math-7B)
  RUNPOD_TRUST_BASE=0|1
  RUNPOD_TRUST_FINETUNED=0|1

Backend on your laptop/EC2:
  export MODEL_BASE_URL=http://<runpod-public-ip>:<public-port>
  # no trailing slash; uvicorn default port 8000 if you map 8000->public

Explainability routes return empty payloads so the UI does not 500 until you
implement real hooks on the pod.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="RunPod HF bridge", version="0.1.0")

_cache: Dict[str, Tuple[Any, Any]] = {}


def _trust(name: str) -> bool:
    return os.getenv(name, "0").strip() in ("1", "true", "True", "yes")


def _dtype():
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def _load(path: str, trust: bool):
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


class PromptBody(BaseModel):
    prompt: str
    max_new_tokens: Optional[int] = 512


@app.get("/health")
def health():
    return {"status": "ok"}


def _generate(path: str, trust: bool, body: PromptBody) -> Dict[str, Any]:
    tok, model = _load(path, trust)
    max_new = min(int(body.max_new_tokens or 512), 4096)
    pad = tok.pad_token_id or tok.eos_token_id
    inputs = tok(body.prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=False,
            pad_token_id=pad,
        )
    text = tok.decode(out[0], skip_special_tokens=True)
    return {"response": text}


@app.post("/base")
def gen_base(body: PromptBody):
    path = os.getenv("RUNPOD_PATH_BASE", "").strip()
    if not path:
        raise HTTPException(status_code=500, detail="RUNPOD_PATH_BASE not set")
    return _generate(path, _trust("RUNPOD_TRUST_BASE"), body)


@app.post("/finetuned")
def gen_ft(body: PromptBody):
    path = os.getenv("RUNPOD_PATH_FINETUNED", "").strip()
    if not path:
        raise HTTPException(status_code=500, detail="RUNPOD_PATH_FINETUNED not set")
    return _generate(path, _trust("RUNPOD_TRUST_FINETUNED"), body)


# --- Stubs matching remote_model_client explain calls (implement later) ---


class ExplainBody(BaseModel):
    model_type: str
    prompt: str
    response: Optional[str] = None


@app.post("/explain/confidence")
def ex_confidence(_: ExplainBody):
    return {"token_confidence": []}


@app.post("/explain/logit-lens")
def ex_logit(_: ExplainBody):
    return {"logit_lens": []}


@app.post("/explain/hidden-states")
def ex_hidden(_: ExplainBody):
    return {"hidden_state_norms": []}


@app.post("/explain/attribution")
def ex_attr(_: ExplainBody):
    return {"gradient_attribution": []}


@app.post("/explain/attention")
def ex_attn(_: ExplainBody):
    return {"tokens": [], "matrix": [], "layer": 0, "head": 0}
