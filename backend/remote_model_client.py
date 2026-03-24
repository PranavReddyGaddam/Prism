import os
import httpx
from typing import Dict, Optional

# Get ngrok URL from environment variable
MODEL_BASE_URL = os.getenv("MODEL_BASE_URL", "")

# Required header to skip ngrok browser warning page
NGROK_HEADERS = {"ngrok-skip-browser-warning": "true"}

# Timeout for model inference (120 seconds as recommended)
TIMEOUT = 120.0


def format_math_prompt(problem: str) -> str:
    """Format prompt in the PRM dataset format that the fine-tuned model expects."""
    return f"### Problem:\n{problem}\n### Solution:\n"


async def get_base_gemma_response(prompt: str, max_new_tokens: int = 512) -> Dict:
    """
    Get response from base Gemma 3 4B Pretrained model (google/gemma-3-4b-pt) via ngrok tunnel.
    
    Args:
        prompt: The input prompt (will be formatted for math problems)
        max_new_tokens: Maximum tokens to generate (default 512)
    
    Returns:
        Dict with 'response' key containing generated text
    """
    if not MODEL_BASE_URL:
        raise ValueError("MODEL_BASE_URL environment variable not set")
    
    formatted_prompt = format_math_prompt(prompt)
    
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        response = await client.post(
            f"{MODEL_BASE_URL}/base",
            json={"prompt": formatted_prompt},
            headers=NGROK_HEADERS
        )
        response.raise_for_status()
        return response.json()


async def get_finetuned_gemma_response(prompt: str, max_new_tokens: int = 512) -> Dict:
    """
    Get response from fine-tuned Gemma 3 4B + LoRA model (PRM math dataset) via ngrok tunnel.
    
    Model: google/gemma-3-4b-pt with LoRA adapter (rank=16, alpha=32)
    Training: Fine-tuned on PRM (Process Reward Model) math dataset
    
    Args:
        prompt: The input prompt (will be formatted for math problems)
        max_new_tokens: Maximum tokens to generate (default 512)
    
    Returns:
        Dict with 'response' key containing generated text
    """
    if not MODEL_BASE_URL:
        raise ValueError("MODEL_BASE_URL environment variable not set")
    
    formatted_prompt = format_math_prompt(prompt)
    
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        response = await client.post(
            f"{MODEL_BASE_URL}/finetuned",
            json={"prompt": formatted_prompt},
            headers=NGROK_HEADERS
        )
        response.raise_for_status()
        return response.json()


async def get_model_comparison(prompt: str) -> Dict:
    """
    Get responses from both base and fine-tuned models for comparison.
    
    Args:
        prompt: The input prompt
    
    Returns:
        Dict with 'base_response' and 'finetuned_response' keys
    """
    base_result = await get_base_gemma_response(prompt)
    finetuned_result = await get_finetuned_gemma_response(prompt)
    
    return {
        "problem": prompt,
        "base_response": base_result["response"],
        "finetuned_response": finetuned_result["response"]
    }


async def check_remote_model_health() -> Dict:
    """
    Check health status of remote models.
    
    Returns:
        Dict with health status information
    """
    if not MODEL_BASE_URL:
        return {"status": "error", "message": "MODEL_BASE_URL not configured"}
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{MODEL_BASE_URL}/health",
                headers=NGROK_HEADERS
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ============================================================================
# EXPLAINABILITY FUNCTIONS
# ============================================================================

async def get_token_confidence(model_type: str, prompt: str, response: str) -> Dict:
    """
    Get token-by-token confidence scores from remote model.
    
    Args:
        model_type: "base" or "finetuned"
        prompt: The formatted prompt
        response: The generated response
    
    Returns:
        Dict with token_confidence list
    """
    if not MODEL_BASE_URL:
        raise ValueError("MODEL_BASE_URL environment variable not set")
    
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        response_data = await client.post(
            f"{MODEL_BASE_URL}/explain/confidence",
            json={
                "model_type": model_type,
                "prompt": prompt,
                "response": response
            },
            headers=NGROK_HEADERS
        )
        response_data.raise_for_status()
        return response_data.json()


async def get_attention_weights(model_type: str, prompt: str, attn_layer: int = 0, attn_head: int = 0) -> Dict:
    """
    Get attention weights for specific layer and head from remote model.
    
    Args:
        model_type: "base" or "finetuned"
        prompt: The user prompt (not formatted)
        attn_layer: Layer index
        attn_head: Head index
    
    Returns:
        Dict with tokens, matrix, layer, head
    """
    if not MODEL_BASE_URL:
        raise ValueError("MODEL_BASE_URL environment variable not set")
    
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        response = await client.post(
            f"{MODEL_BASE_URL}/explain/attention",
            json={
                "model_type": model_type,
                "prompt": prompt,
                "attn_layer": attn_layer,
                "attn_head": attn_head
            },
            headers=NGROK_HEADERS
        )
        response.raise_for_status()
        return response.json()


async def get_logit_lens(model_type: str, prompt: str, response: str) -> Dict:
    """
    Get logit lens data (predicted tokens at each layer) from remote model.
    
    Args:
        model_type: "base" or "finetuned"
        prompt: The formatted prompt
        response: The generated response
    
    Returns:
        Dict with logit_lens list
    """
    if not MODEL_BASE_URL:
        raise ValueError("MODEL_BASE_URL environment variable not set")
    
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        response_data = await client.post(
            f"{MODEL_BASE_URL}/explain/logit-lens",
            json={
                "model_type": model_type,
                "prompt": prompt,
                "response": response
            },
            headers=NGROK_HEADERS
        )
        response_data.raise_for_status()
        return response_data.json()


async def get_hidden_states(model_type: str, prompt: str) -> Dict:
    """
    Get hidden state norms at each layer from remote model.
    
    Args:
        model_type: "base" or "finetuned"
        prompt: The formatted prompt
    
    Returns:
        Dict with hidden_state_norms list
    """
    if not MODEL_BASE_URL:
        raise ValueError("MODEL_BASE_URL environment variable not set")
    
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        response = await client.post(
            f"{MODEL_BASE_URL}/explain/hidden-states",
            json={
                "model_type": model_type,
                "prompt": prompt
            },
            headers=NGROK_HEADERS
        )
        response.raise_for_status()
        return response.json()


async def get_gradient_attribution(model_type: str, prompt: str, response: str) -> Dict:
    """
    Get gradient-based attribution scores from remote model.
    
    Args:
        model_type: "base" or "finetuned"
        prompt: The user prompt (not formatted)
        response: The generated response
    
    Returns:
        Dict with gradient_attribution list
    """
    if not MODEL_BASE_URL:
        raise ValueError("MODEL_BASE_URL environment variable not set")
    
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        response_data = await client.post(
            f"{MODEL_BASE_URL}/explain/attribution",
            json={
                "model_type": model_type,
                "prompt": prompt,
                "response": response
            },
            headers=NGROK_HEADERS
        )
        response_data.raise_for_status()
        return response_data.json()
