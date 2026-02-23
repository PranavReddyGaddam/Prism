import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Base directory: prefer a sibling models/ folder (cloud), fall back to local dev paths
_HERE = os.path.dirname(os.path.abspath(__file__))
_MODELS_DIR = os.path.join(_HERE, "..", "models")


def _model_path(cloud_name: str, local_fallback: str) -> str:
    """Return cloud path if it exists, otherwise local dev path."""
    cloud = os.path.abspath(os.path.join(_MODELS_DIR, cloud_name))
    return cloud if os.path.isdir(cloud) else local_fallback


MODEL_CONFIGS = {
    "phi3": {
        "base_path": _model_path(
            "phi-3-mini-4k-instruct",
            "/Users/pranav/Projects/Major Project/phi-3-mini-4k-instruct",
        ),
        "adapter_path": _model_path(
            "phi3_mmlu_fever_hotpot_lora_adapter",
            "/Users/pranav/Projects/Major Project/phi3_mmlu_fever_hotpot_lora_adapter",
        ),
        "dtype": torch.float16,
    },
    # "deepseek": {
    #     "base_path": _model_path(
    #         "deepseek-model",
    #         "/Users/pranav/Projects/Major Project/deepseek-model",
    #     ),
    #     "adapter_path": None,
    #     "dtype": torch.bfloat16,
    # },
}

_model_cache: dict = {}


def get_model(model_id: str):
    if model_id not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model_id: {model_id}. Choose from {list(MODEL_CONFIGS.keys())}")

    if model_id in _model_cache:
        return _model_cache[model_id]

    config = MODEL_CONFIGS[model_id]
    dtype = config["dtype"]

    tokenizer = AutoTokenizer.from_pretrained(config["base_path"])

    model = AutoModelForCausalLM.from_pretrained(
        config["base_path"],
        torch_dtype=dtype,
        device_map="auto",  # Will use GPU if available, otherwise CPU
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )

    # Detect device for tensor operations
    device = next(model.parameters()).device

    if model_id == "phi3":
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if config["adapter_path"]:
            # The LoRA adapter was trained on vocab size 32012; resize before loading
            model.resize_token_embeddings(32012)
            model = PeftModel.from_pretrained(model, config["adapter_path"])
            model = model.merge_and_unload()

    model.eval()
    _model_cache[model_id] = (model, tokenizer)
    return model, tokenizer


def _format_prompt(model_id: str, tokenizer, prompt: str) -> str:
    if model_id == "deepseek":
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        return (
            "### System:\n"
            "You are a helpful assistant. Explain step-by-step in detail, "
            "then give the final answer. Follow the required format: Final: ...\n\n"
            f"### User:\n{prompt}\n\n### Assistant:\n"
        )


def run_inference(model_id: str, prompt: str, max_new_tokens: int = 512) -> dict:
    model, tokenizer = get_model(model_id)

    formatted = _format_prompt(model_id, tokenizer, prompt)
    inputs = tokenizer(formatted, return_tensors="pt").to(device)
    input_length = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][input_length:]
    full_response = tokenizer.decode(new_tokens, skip_special_tokens=False)

    eos = tokenizer.eos_token or "<｜end▁of▁sentence｜>"
    full_response = full_response.replace(eos, "").strip()

    for stop in ["### User:", "<｜User｜>", "<｜end▁of▁sentence｜>"]:
        if stop in full_response:
            full_response = full_response.split(stop)[0].strip()

    thinking = None
    final_answer = None

    if model_id == "deepseek":
        if "</think>" in full_response:
            parts = full_response.split("</think>", 1)
            thinking = parts[0].strip()
            final_answer = parts[1].strip()
        elif "<think>" in full_response:
            thinking = full_response.split("<think>", 1)[1].strip()
            final_answer = None
        else:
            final_answer = full_response.strip()
    else:
        if "Final:" in full_response:
            final_answer = full_response.split("Final:")[-1].strip()
        else:
            final_answer = full_response.strip()

    return {
        "response": full_response,
        "thinking": thinking,
        "final_answer": final_answer,
        "token_count": len(new_tokens),
    }
