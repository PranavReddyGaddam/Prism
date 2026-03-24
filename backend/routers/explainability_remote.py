"""
Explainability router for remote Gemma models via Colab.
This router calls the Colab Flask server's /explain endpoints.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import asyncio

router = APIRouter()


class ExplainRequest(BaseModel):
    model_id: str  # "gemma-base" or "gemma-finetuned"
    prompt: str
    max_new_tokens: int = 256
    attn_layer: int = 0
    attn_head: int = 0
    response: Optional[str] = None  # pass pre-generated response to skip re-running inference


class TokenConfidence(BaseModel):
    token: str
    confidence: float


class AttentionData(BaseModel):
    tokens: list[str]
    matrix: list[list[float]]
    layer: int
    head: int


class LogitLensLayer(BaseModel):
    layer: int
    predicted_token: str
    probability: float


class GradientAttribution(BaseModel):
    token: str
    score: float


class HiddenStateNorm(BaseModel):
    layer: int
    norm: float


def get_model_type(model_id: str) -> str:
    """Map model_id to Colab model_type."""
    if model_id == "gemma-base":
        return "base"
    elif model_id == "gemma-finetuned":
        return "finetuned"
    else:
        raise ValueError(f"Unknown model_id: {model_id}")


@router.post("/attention", response_model=AttentionData)
async def get_attention(request: ExplainRequest):
    """Get attention weights for specific layer and head."""
    try:
        from remote_model_client import get_attention_weights
        
        model_type = get_model_type(request.model_id)
        
        result = await get_attention_weights(
            model_type=model_type,
            prompt=request.prompt,
            attn_layer=request.attn_layer,
            attn_head=request.attn_head
        )
        
        return AttentionData(
            tokens=result["tokens"],
            matrix=result["matrix"],
            layer=result["layer"],
            head=result["head"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/confidence")
async def get_token_confidence_endpoint(request: ExplainRequest):
    """Get token-by-token confidence scores."""
    try:
        from remote_model_client import get_token_confidence, format_math_prompt
        
        model_type = get_model_type(request.model_id)
        
        # If no response provided, generate one first
        if not request.response:
            from remote_model_client import get_base_gemma_response, get_finetuned_gemma_response
            
            if model_type == "base":
                gen_result = await get_base_gemma_response(request.prompt, request.max_new_tokens)
            else:
                gen_result = await get_finetuned_gemma_response(request.prompt, request.max_new_tokens)
            
            response_text = gen_result["response"]
        else:
            response_text = request.response
        
        # Format prompt for Gemma
        formatted_prompt = format_math_prompt(request.prompt)
        
        result = await get_token_confidence(
            model_type=model_type,
            prompt=formatted_prompt,
            response=response_text
        )
        
        return {"token_confidence": result["token_confidence"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/logit-lens")
async def get_logit_lens_endpoint(request: ExplainRequest):
    """Get logit lens data (predicted tokens at each layer)."""
    try:
        from remote_model_client import get_logit_lens, format_math_prompt
        
        model_type = get_model_type(request.model_id)
        
        # If no response provided, generate one first
        if not request.response:
            from remote_model_client import get_base_gemma_response, get_finetuned_gemma_response
            
            if model_type == "base":
                gen_result = await get_base_gemma_response(request.prompt, request.max_new_tokens)
            else:
                gen_result = await get_finetuned_gemma_response(request.prompt, request.max_new_tokens)
            
            response_text = gen_result["response"]
        else:
            response_text = request.response
        
        # Format prompt for Gemma
        formatted_prompt = format_math_prompt(request.prompt)
        
        result = await get_logit_lens(
            model_type=model_type,
            prompt=formatted_prompt,
            response=response_text
        )
        
        return {"logit_lens": result["logit_lens"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/hidden-states")
async def get_hidden_states_endpoint(request: ExplainRequest):
    """Get hidden state norms at each layer."""
    try:
        from remote_model_client import get_hidden_states, format_math_prompt
        
        model_type = get_model_type(request.model_id)
        
        # Format prompt for Gemma
        formatted_prompt = format_math_prompt(request.prompt)
        
        result = await get_hidden_states(
            model_type=model_type,
            prompt=formatted_prompt
        )
        
        return {"hidden_state_norms": result["hidden_state_norms"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/attribution")
async def get_gradient_attribution_endpoint(request: ExplainRequest):
    """Get gradient-based attribution scores."""
    try:
        from remote_model_client import get_gradient_attribution
        
        model_type = get_model_type(request.model_id)
        
        # If no response provided, generate one first
        if not request.response:
            from remote_model_client import get_base_gemma_response, get_finetuned_gemma_response
            
            if model_type == "base":
                gen_result = await get_base_gemma_response(request.prompt, request.max_new_tokens)
            else:
                gen_result = await get_finetuned_gemma_response(request.prompt, request.max_new_tokens)
            
            response_text = gen_result["response"]
        else:
            response_text = request.response
        
        result = await get_gradient_attribution(
            model_type=model_type,
            prompt=request.prompt,
            response=response_text
        )
        
        return {"gradient_attribution": result["gradient_attribution"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
