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
from typing import Any, Dict, Optional, Tuple

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="RunPod HF bridge — math models", version="0.2.0")

# Model cache: path → (tokenizer, model)
_cache: Dict[str, Tuple[Any, Any]] = {}

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
    return {"status": "ok", "models_configured": configured}


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

    tok, model = _load(path, _trust(trust_env))
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
    return {"model_id": body.model_id, "response": text}


# --- Explainability stubs — return empty payloads until implemented ---

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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
