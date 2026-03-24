from fastapi import APIRouter
import asyncio
from remote_model_client import check_remote_model_health

router = APIRouter()

AVAILABLE_MODELS = [
    {
        "id": "gemma-base",
        "name": "Gemma 3 4B Pretrained (via Colab)",
        "path": "google/gemma-3-4b-pt",
        "parameters": "4B",
        "type": "pretrained-base",
        "location": "remote",
        "context_window": "128K",
        "developer": "Google DeepMind",
    },
    {
        "id": "gemma-finetuned",
        "name": "Gemma 3 4B + LoRA PRM Math (via Colab)",
        "path": "google/gemma-3-4b-pt",
        "parameters": "4B",
        "type": "math-reasoning",
        "location": "remote",
        "context_window": "128K",
        "adapter": "LoRA (rank=16, alpha=32)",
        "dataset": "PRM Math",
    },
]


@router.get("/")
def list_models():
    return {"models": AVAILABLE_MODELS}


@router.get("/{model_id}")
def get_model(model_id: str):
    model = next((m for m in AVAILABLE_MODELS if m["id"] == model_id), None)
    if model is None:
        return {"error": f"Model '{model_id}' not found"}
    return model


@router.get("/health/remote")
def check_remote_health():
    """Check health status of remote Colab models via ngrok tunnel."""
    return asyncio.run(check_remote_model_health())
