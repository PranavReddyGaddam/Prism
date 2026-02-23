from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from model_manager import run_inference

router = APIRouter()


class GenerationRequest(BaseModel):
    model_id: str
    prompt: str
    max_new_tokens: int = 512


class GenerationResponse(BaseModel):
    model_id: str
    prompt: str
    response: str
    thinking: Optional[str] = None
    final_answer: Optional[str] = None
    token_count: int


@router.post("/", response_model=GenerationResponse)
def generate(request: GenerationRequest):
    try:
        result = run_inference(request.model_id, request.prompt, request.max_new_tokens)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return GenerationResponse(
        model_id=request.model_id,
        prompt=request.prompt,
        response=result["response"],
        thinking=result["thinking"],
        final_answer=result["final_answer"],
        token_count=result["token_count"],
    )
