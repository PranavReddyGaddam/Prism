
MODEL_CONFIGS = {
    "qwen-2.5-math-7b": {
        "description": "Qwen2.5-Math-7B — math reasoning",
        "parameters": "7B",
        "hf_repo": "Qwen/Qwen2.5-Math-7B",
        "pod_dir": "qwen25-math-7b",
    },
    "deepseek-math-7b": {
        "description": "DeepSeek-Math-7B-Instruct — math instruction following",
        "parameters": "7B",
        "hf_repo": "deepseek-ai/deepseek-math-7b-instruct",
        "pod_dir": "deepseek-math-7b",
    },
    "internlm2-math-7b": {
        "description": "InternLM2-Math-Plus-7B — advanced math reasoning",
        "parameters": "7B",
        "hf_repo": "internlm/internlm2-math-plus-7b",
        "pod_dir": "internlm2-math-plus-7b",
    },
    "wizardmath-7b": {
        "description": "WizardMath-7B-V1.1 — math problem solving",
        "parameters": "7B",
        "hf_repo": "WizardLMTeam/WizardMath-7B-V1.1",
        "pod_dir": "wizardmath-7b-v1.1",
    },
}


def run_inference(model_id: str, prompt: str, max_new_tokens: int = 512) -> dict:
    if model_id not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model_id: {model_id!r}. Valid: {list(MODEL_CONFIGS.keys())}"
        )

    import asyncio
    from remote_model_client import get_model_response

    result = asyncio.run(get_model_response(model_id, prompt, max_new_tokens))
    full_response = result.get("response", "")

    # Strip echoed prompt header if present
    if "### Solution:" in full_response:
        body = full_response.split("### Solution:")[-1].strip()
    else:
        body = full_response.strip()

    return {
        "response": full_response,
        "thinking": body,
        "final_answer": body,
        "token_count": len(full_response) // 4,
    }
