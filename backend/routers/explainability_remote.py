"""
Explainability router — proxies to the RunPod pod's /explain/* stubs.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import asyncio

from model_manager import MODEL_CONFIGS

router = APIRouter()

VALID_MODEL_IDS = set(MODEL_CONFIGS.keys())


class ExplainRequest(BaseModel):
    model_id: str
    prompt: str
    max_new_tokens: int = 256
    attn_layer: int = 0
    attn_head: int = 0
    response: Optional[str] = None


class AttentionData(BaseModel):
    tokens: list[str]
    matrix: list[list[float]]
    layer: int
    head: int


def _validate(model_id: str) -> None:
    if model_id not in VALID_MODEL_IDS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model_id {model_id!r}. Valid: {sorted(VALID_MODEL_IDS)}",
        )


@router.post("/attention", response_model=AttentionData)
async def get_attention(request: ExplainRequest):
    _validate(request.model_id)
    try:
        from remote_model_client import get_attention_weights
        result = await get_attention_weights(
            model_id=request.model_id,
            prompt=request.prompt,
            attn_layer=request.attn_layer,
            attn_head=request.attn_head,
        )
        return AttentionData(
            tokens=result["tokens"],
            matrix=result["matrix"],
            layer=result["layer"],
            head=result["head"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/confidence")
async def get_token_confidence_endpoint(request: ExplainRequest):
    _validate(request.model_id)
    try:
        from remote_model_client import get_token_confidence, get_model_response, format_math_prompt

        response_text = request.response
        if not response_text:
            gen = await get_model_response(request.model_id, request.prompt, request.max_new_tokens)
            response_text = gen["response"]

        result = await get_token_confidence(
            model_id=request.model_id,
            prompt=format_math_prompt(request.prompt),
            response=response_text,
        )
        return {"token_confidence": result["token_confidence"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/logit-lens")
async def get_logit_lens_endpoint(request: ExplainRequest):
    _validate(request.model_id)
    try:
        from remote_model_client import get_logit_lens, get_model_response, format_math_prompt

        response_text = request.response
        if not response_text:
            gen = await get_model_response(request.model_id, request.prompt, request.max_new_tokens)
            response_text = gen["response"]

        result = await get_logit_lens(
            model_id=request.model_id,
            prompt=format_math_prompt(request.prompt),
            response=response_text,
        )
        return {"logit_lens": result["logit_lens"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/hidden-states")
async def get_hidden_states_endpoint(request: ExplainRequest):
    _validate(request.model_id)
    try:
        from remote_model_client import get_hidden_states, format_math_prompt
        result = await get_hidden_states(
            model_id=request.model_id,
            prompt=format_math_prompt(request.prompt),
        )
        return {"hidden_state_norms": result["hidden_state_norms"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/attribution")
async def get_gradient_attribution_endpoint(request: ExplainRequest):
    _validate(request.model_id)
    try:
        from remote_model_client import get_gradient_attribution, get_model_response

        response_text = request.response
        if not response_text:
            gen = await get_model_response(request.model_id, request.prompt, request.max_new_tokens)
            response_text = gen["response"]

        result = await get_gradient_attribution(
            model_id=request.model_id,
            prompt=request.prompt,
            response=response_text,
        )
        return {
            "gradient_attribution": result.get("gradient_attribution", []),
            "position_predictions": result.get("position_predictions", []),
            "layer_attention": result.get("layer_attention", []),
            "tokens": result.get("tokens", []),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
