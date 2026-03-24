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


class ComparisonRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 512


class ComparisonResponse(BaseModel):
    prompt: str
    base_response: str
    finetuned_response: str
    base_token_count: int
    finetuned_token_count: int


@router.post("/", response_model=GenerationResponse)
def generate(request: GenerationRequest):
    try:
        result = run_inference(request.model_id, request.prompt, request.max_new_tokens)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback
        error_detail = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        print(f"Generation error: {error_detail}")
        raise HTTPException(status_code=500, detail=str(e))

    return GenerationResponse(
        model_id=request.model_id,
        prompt=request.prompt,
        response=result["response"],
        thinking=result["thinking"],
        final_answer=result["final_answer"],
        token_count=result["token_count"],
    )


@router.post("/compare", response_model=ComparisonResponse)
def compare_models(request: ComparisonRequest):
    """Compare base Gemma and fine-tuned Gemma models side-by-side."""
    try:
        # Get responses from both models
        base_result = run_inference("gemma-base", request.prompt, request.max_new_tokens)
        finetuned_result = run_inference("gemma-finetuned", request.prompt, request.max_new_tokens)
        
        return ComparisonResponse(
            prompt=request.prompt,
            base_response=base_result["response"],
            finetuned_response=finetuned_result["response"],
            base_token_count=base_result["token_count"],
            finetuned_token_count=finetuned_result["token_count"],
        )
    except Exception as e:
        import traceback
        error_detail = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        print(f"Comparison error: {error_detail}")
        raise HTTPException(status_code=500, detail=str(e))
