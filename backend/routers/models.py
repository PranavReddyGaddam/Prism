from fastapi import APIRouter

router = APIRouter()

AVAILABLE_MODELS = [
    {
        "id": "deepseek",
        "name": "DeepSeek-R1-Distill-Qwen-1.5B",
        "path": "../deepseek-model",
        "parameters": "1.78B",
        "type": "chain-of-thought",
    },
    {
        "id": "phi3",
        "name": "Microsoft Phi-3-mini-4k-instruct",
        "path": "../phi-3-mini-4k-instruct",
        "parameters": "3.8B",
        "type": "instruction-tuned",
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
