from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from model_manager import MODEL_CONFIGS, run_inference

router = APIRouter()

VALID_MODEL_IDS = set(MODEL_CONFIGS.keys())


class GenerationRequest(BaseModel):
    model_id: str
    prompt: str
    max_new_tokens: int = 256


class GenerationResponse(BaseModel):
    model_id: str
    prompt: str
    response: str
    thinking: Optional[str] = None
    final_answer: Optional[str] = None
    token_count: int


@router.post("/", response_model=GenerationResponse)
def generate(request: GenerationRequest):
    if request.model_id not in VALID_MODEL_IDS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model_id {request.model_id!r}. Valid: {sorted(VALID_MODEL_IDS)}",
        )
    try:
        result = run_inference(request.model_id, request.prompt, request.max_new_tokens)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback
        print(f"Generation error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

    return GenerationResponse(
        model_id=request.model_id,
        prompt=request.prompt,
        response=result["response"],
        thinking=result["thinking"],
        final_answer=result["final_answer"],
        token_count=result["token_count"],
    )
