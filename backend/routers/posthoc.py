"""
Post-hoc analysis router — proxies to the pod's /posthoc/* endpoints.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from model_manager import MODEL_CONFIGS

router = APIRouter()

VALID_MODEL_IDS = set(MODEL_CONFIGS.keys())


class PostHocRequest(BaseModel):
    model_id: str
    prompt: str
    response: Optional[str] = None
    n_samples: Optional[int] = None


def _validate(model_id: str) -> None:
    if model_id not in VALID_MODEL_IDS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model_id {model_id!r}. Valid: {sorted(VALID_MODEL_IDS)}",
        )


@router.post("/lime")
async def lime_endpoint(request: PostHocRequest):
    _validate(request.model_id)
    try:
        from remote_model_client import get_posthoc_lime, format_math_prompt
        result = await get_posthoc_lime(
            model_id=request.model_id,
            prompt=format_math_prompt(request.prompt),
            response=request.response or "",
            n_samples=request.n_samples or 60,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tokenshap")
async def tokenshap_endpoint(request: PostHocRequest):
    _validate(request.model_id)
    try:
        from remote_model_client import get_posthoc_tokenshap, format_math_prompt
        result = await get_posthoc_tokenshap(
            model_id=request.model_id,
            prompt=format_math_prompt(request.prompt),
            response=request.response or "",
            n_samples=request.n_samples or 50,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/counterfactual")
async def counterfactual_endpoint(request: PostHocRequest):
    _validate(request.model_id)
    try:
        from remote_model_client import get_posthoc_counterfactual, format_math_prompt
        result = await get_posthoc_counterfactual(
            model_id=request.model_id,
            prompt=format_math_prompt(request.prompt),
            response=request.response or "",
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
