import math
import torch
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from model_manager import get_model, run_inference, _format_prompt

router = APIRouter()


def _safe_float(v: float) -> float:
    if math.isnan(v) or math.isinf(v):
        return 0.0
    return v


def _safe_matrix(matrix: list) -> list:
    return [[_safe_float(v) for v in row] for row in matrix]


class ExplainRequest(BaseModel):
    model_id: str
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


class ExplainResponse(BaseModel):
    model_id: str
    prompt: str
    response: str
    thinking: Optional[str] = None
    final_answer: Optional[str] = None
    token_confidence: list[TokenConfidence]
    attention: Optional[AttentionData]
    logit_lens: list[LogitLensLayer]
    gradient_attribution: list[GradientAttribution]
    hidden_state_norms: list[HiddenStateNorm]


@router.post("/attention", response_model=AttentionData)
def get_attention(request: ExplainRequest):
    try:
        model, tokenizer = get_model(request.model_id)
        device = next(model.parameters()).device
        
        # Tokenize just the user prompt
        user_inputs = tokenizer(request.prompt, return_tensors="pt").to(device)
        user_tokens = tokenizer.convert_ids_to_tokens(user_inputs["input_ids"][0])

        with torch.no_grad():
            out = model(**user_inputs, output_attentions=True)
        layer_attn = out.attentions[request.attn_layer][0, request.attn_head]

        return AttentionData(
            tokens=user_tokens,
            matrix=_safe_matrix(layer_attn.float().tolist()),
            layer=request.attn_layer,
            head=request.attn_head,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/confidence")
def get_token_confidence(request: ExplainRequest):
    try:
        model, tokenizer = get_model(request.model_id)
        device = next(model.parameters()).device
        response_text = request.response if request.response else run_inference(request.model_id, request.prompt, request.max_new_tokens)["response"]
        formatted = _format_prompt(request.model_id, tokenizer, request.prompt)
        inputs = tokenizer(formatted, return_tensors="pt").to(device)
        full_ids = tokenizer(formatted + response_text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**full_ids)
        logits = out.logits[0]
        probs = torch.softmax(logits, dim=-1)
        input_len = inputs["input_ids"].shape[1]
        gen_ids = full_ids["input_ids"][0][input_len:]
        gen_tokens = tokenizer.convert_ids_to_tokens(gen_ids)
        token_confidence = [
            {"token": tok, "confidence": _safe_float(float(probs[input_len + i - 1, gen_ids[i]].item()))}
            for i, tok in enumerate(gen_tokens)
        ]
        return {"token_confidence": token_confidence}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/logit-lens")
def get_logit_lens(request: ExplainRequest):
    try:
        model, tokenizer = get_model(request.model_id)
        device = next(model.parameters()).device
        response_text = request.response if request.response else run_inference(request.model_id, request.prompt, request.max_new_tokens)["response"]
        
        # Get response tokens
        response_ids = tokenizer(response_text, return_tensors="pt")["input_ids"][0]
        
        # Sample every 5th word (including first word)
        sampled_positions = list(range(0, len(response_ids), 5))
        
        logit_lens = []
        for pos in sampled_positions:
            # Get prefix up to this position
            prefix_text = tokenizer.decode(response_ids[:pos]) if pos > 0 else ""
            full_input = request.prompt + prefix_text
            
            # Get model outputs for this position
            inputs = tokenizer(_format_prompt(request.model_id, tokenizer, full_input), return_tensors="pt").to(device)
            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True)
            
            lm_head = model.lm_head if hasattr(model, 'lm_head') else model.base_model.lm_head
            
            for layer_idx, hs in enumerate(out.hidden_states):
                last_hidden = hs[0, -1, :].unsqueeze(0)
                with torch.no_grad():
                    layer_logits = lm_head(last_hidden).float()
                layer_probs = torch.softmax(layer_logits, dim=-1)
                
                # Get actual token at this position
                actual_token_id = response_ids[pos] if pos < len(response_ids) else response_ids[-1]
                actual_token = tokenizer.decode([actual_token_id])
                actual_prob = layer_probs[0, actual_token_id].item()
                
                logit_lens.append({
                    "layer": layer_idx, 
                    "word_position": pos,
                    "predicted_token": actual_token,
                    "probability": _safe_float(float(actual_prob))
                })
        
        return {"logit_lens": logit_lens}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/hidden-states")
def get_hidden_states(request: ExplainRequest):
    try:
        model, tokenizer = get_model(request.model_id)
        device = next(model.parameters()).device
        formatted = _format_prompt(request.model_id, tokenizer, request.prompt)
        inputs = tokenizer(formatted, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        hidden_state_norms = [
            {"layer": i, "norm": _safe_float(float(hs[0].float().norm(dim=-1).mean().item()))}
            for i, hs in enumerate(out.hidden_states)
        ]
        return {"hidden_state_norms": hidden_state_norms}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/attribution")
def get_gradient_attribution(request: ExplainRequest):
    try:
        model, tokenizer = get_model(request.model_id)
        device = next(model.parameters()).device
        response_text = request.response if request.response else run_inference(request.model_id, request.prompt, request.max_new_tokens)["response"]
        
        # Tokenize just the user prompt (no system prefix)
        user_inputs = tokenizer(request.prompt, return_tensors="pt").to(device)
        user_tokens = tokenizer.convert_ids_to_tokens(user_inputs["input_ids"][0])
        
        # Get the first generated token as target
        full_inputs = tokenizer(request.prompt + response_text, return_tensors="pt").to(device)
        target_id = full_inputs["input_ids"][0, user_inputs["input_ids"].shape[1]].item()
        
        model.zero_grad()
        embed = model.get_input_embeddings()(user_inputs["input_ids"]).detach().requires_grad_(True)
        with torch.enable_grad():
            out = model(inputs_embeds=embed, attention_mask=user_inputs.get("attention_mask"))
            loss = out.logits[0, -1, target_id]
            loss.backward()
        grads = embed.grad[0].float().norm(dim=-1)
        
        gradient_attribution = [
            {"token": tok, "score": _safe_float(float(grads[i].item()))}
            for i, tok in enumerate(user_tokens)
        ]
        return {"gradient_attribution": gradient_attribution}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
