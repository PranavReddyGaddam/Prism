# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Prism** is an LLM mechanistic interpretability platform that visualizes what language models are *actually computing* (attention patterns, token confidence, logit lens, gradient attribution, hidden states) rather than just what they say they're thinking (Chain-of-Thought). The tagline is "Inside the Black Box."

## Architecture

The system has three layers:

1. **Frontend** (`frontend/`) — React + Vite + TypeScript SPA. User submits a math prompt; results appear in a tabbed model-comparison bento grid with expandable visualization cards.

2. **Backend** (`backend/`) — FastAPI server that acts as a proxy/orchestrator. It does NOT load models locally — it forwards all inference and explainability requests to a remote GPU server via HTTP.

3. **RunPod GPU server** (`scripts/runpod_hf_server.py`) — FastAPI server deployed on a RunPod GPU instance. Loads HuggingFace models from local snapshot directories and serves `/base`, `/finetuned`, and `/explain/*` endpoints. The backend connects to it via `MODEL_BASE_URL` env var.

### Data flow

```
User prompt → Frontend → Backend (FastAPI, port 8000)
                             ↓
                    remote_model_client.py
                             ↓
                    RunPod HTTP (MODEL_BASE_URL)
                    /base | /finetuned | /explain/*
```

### Current model state

`frontend/src/modelSpecs.ts` defines four planned comparison tabs (Qwen 2.5 Math 7B, DeepSeek Math 7B, InternLM2 Math+ 7B, WizardMath 7B) but the backend `model_manager.py` currently only supports `gemma-base` and `gemma-finetuned`. The frontend's `handleSubmit` currently uses **mock/template data** (`buildTemplateSlot`) — it does not yet call the backend API. The info banner says "Template preview per model — swap for RunPod API when connected."

## Commands

### Frontend
```bash
cd frontend
npm install       # first time
npm run dev       # dev server at http://localhost:5173
npm run build     # tsc + vite build → dist/
npm run lint      # eslint
npm run preview   # preview production build
```

### Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Backend requires `MODEL_BASE_URL` env var pointing to the RunPod server (set in `backend/.env`).

### RunPod server (deploy on GPU instance)
```bash
# Set env vars on RunPod:
# RUNPOD_PATH_BASE=/path/to/model-snapshot
# RUNPOD_PATH_FINETUNED=/path/to/finetuned-snapshot
pip install fastapi uvicorn torch transformers
uvicorn scripts.runpod_hf_server:app --host 0.0.0.0 --port 8000
```

## Backend API Endpoints

| Route | Description |
|---|---|
| `GET /health` | Health check |
| `GET /models/` | List available models |
| `POST /generate/` | Run inference (`model_id`, `prompt`, `max_new_tokens`) |
| `POST /generate/compare` | Compare base vs finetuned side-by-side |
| `POST /explain/confidence` | Token-level softmax confidence scores |
| `POST /explain/attention` | Attention weight matrix for a given layer/head |
| `POST /explain/logit-lens` | Predicted token at each transformer layer |
| `POST /explain/hidden-states` | L2 norm of hidden states per layer |
| `POST /explain/attribution` | Gradient-based attribution scores per input token |

All `/explain/*` routes accept `model_id` (`gemma-base` or `gemma-finetuned`), `prompt`, optional `response` (skip re-generation), `attn_layer`, `attn_head`.

## Key Files

- `backend/remote_model_client.py` — all HTTP calls to the RunPod server; set `MODEL_BASE_URL` env var (includes `ngrok-skip-browser-warning` header for ngrok tunnels)
- `backend/model_manager.py` — model registry (`MODEL_CONFIGS`) and `run_inference()` dispatcher
- `frontend/src/modelSpecs.ts` — single source of truth for which model tabs appear in the UI
- `frontend/src/components/ModelComparisonBento.tsx` — main visualization grid; `SlotState` is the central data type flowing from `App.tsx` down
- `frontend/src/App.tsx` — top-level state: `slots` (per-model results), `handleSubmit` (currently mock)

## Frontend Stack

React 19, TypeScript, Vite 8, Tailwind CSS 3, Framer Motion, Recharts (charts), D3, Three.js / `@react-three/fiber`, GSAP. Path alias `@/` maps to `frontend/src/`.
