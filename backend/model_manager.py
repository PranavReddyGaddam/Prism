import os


MODEL_CONFIGS = {
    "gemma-base": {
        "remote": True,
        "endpoint": "base",
        "description": "Gemma 3 4B Pretrained (google/gemma-3-4b-pt) via Colab ngrok tunnel",
        "parameters": "4B",
        "context_window": "128K",
        "model_name": "google/gemma-3-4b-pt",
    },
    "gemma-finetuned": {
        "remote": True,
        "endpoint": "finetuned",
        "description": "Gemma 3 4B + LoRA (PRM math dataset) via Colab ngrok tunnel",
        "parameters": "4B",
        "context_window": "128K",
        "model_name": "google/gemma-3-4b-pt",
        "adapter": "LoRA (rank=16, alpha=32)",
    },
}



def run_inference(model_id: str, prompt: str, max_new_tokens: int = 512) -> dict:
    """
    Run inference on remote Gemma models via Colab ngrok tunnel.
    
    Args:
        model_id: Either 'gemma-base' or 'gemma-finetuned'
        prompt: The input prompt
        max_new_tokens: Maximum tokens to generate (default 512)
    
    Returns:
        Dict with response, thinking, final_answer, and token_count
    """
    if model_id not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model_id: {model_id}. Choose from {list(MODEL_CONFIGS.keys())}")
    
    # Import here to avoid circular dependency
    import asyncio
    from remote_model_client import get_base_gemma_response, get_finetuned_gemma_response
    
    # Call appropriate remote model endpoint
    if model_id == "gemma-base":
        result = asyncio.run(get_base_gemma_response(prompt, max_new_tokens))
    elif model_id == "gemma-finetuned":
        result = asyncio.run(get_finetuned_gemma_response(prompt, max_new_tokens))
    else:
        raise ValueError(f"Unknown model: {model_id}")
    
    # Parse response from remote model
    full_response = result.get("response", "")
    
    # For Gemma models, extract thinking and final answer if present
    thinking = None
    final_answer = None
    
    # Gemma fine-tuned model uses step-by-step format
    if "### Solution:" in full_response:
        final_answer = full_response.split("### Solution:")[-1].strip()
    else:
        final_answer = full_response.strip()
    
    # Estimate token count (rough approximation: ~4 chars per token)
    token_count = len(full_response) // 4
    
    return {
        "response": full_response,
        "thinking": thinking,
        "final_answer": final_answer,
        "token_count": token_count,
    }
