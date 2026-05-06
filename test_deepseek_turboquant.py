"""
Test DeepSeekMath-7B-Instruct with TurboQuant KV cache compression on Apple M3 MPS.
Compares baseline (float16) vs TurboQuant at 4-bit and 3.5-bit on math problems.
"""

import sys
import time
import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

sys.path.insert(0, "/Users/pranav/Projects/Major Project/turboquant")
from turboquant.cache import TurboQuantCache

MODEL_PATH = "/Users/pranav/Projects/Major Project/models/deepseek-math-7b-instruct"

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

MATH_PROMPTS = [
    "What is the integral of x^2 from 0 to 3? Please reason step by step, and put your final answer within \\boxed{}.",
    "Find all values of x such that x^2 - 5x + 6 = 0. Please reason step by step, and put your final answer within \\boxed{}.",
]

def format_bytes(b):
    if b == 0:
        return "N/A"
    if b < 1024 * 1024:
        return f"{b / 1024:.1f} KB"
    return f"{b / (1024 * 1024):.2f} MB"

def load_model():
    print(f"Loading DeepSeekMath-7B-Instruct from {MODEL_PATH}...")
    print(f"Device: {DEVICE}\n")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.float16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    model = model.to(DEVICE)
    model.generation_config = GenerationConfig.from_pretrained(MODEL_PATH)
    model.generation_config.pad_token_id = model.generation_config.eos_token_id
    model.eval()

    head_dim = model.config.hidden_size // model.config.num_attention_heads
    print(f"Loaded: {model.config.num_hidden_layers} layers, "
          f"{model.config.num_attention_heads} heads, head_dim={head_dim}")
    print(f"KV heads: {model.config.num_key_value_heads}\n")

    return model, tokenizer

def format_prompt(prompt):
    """DeepSeekMath uses manual User/Assistant format per their model card."""
    return f"User: {prompt}\n\nAssistant:"

def run_baseline(model, tokenizer, prompt, max_new_tokens=100):
    input_text = format_prompt(prompt)
    enc = tokenizer(input_text, return_tensors="pt")
    input_ids = enc.input_ids.to(DEVICE)
    attention_mask = enc.attention_mask.to(DEVICE)

    gc.collect()
    if DEVICE.type == "mps":
        torch.mps.empty_cache()

    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            return_dict_in_generate=True,
        )
    elapsed = time.time() - start

    generated = outputs.sequences[0][input_ids.shape[1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return text, elapsed, generated.shape[0]

def run_turboquant(model, tokenizer, prompt, bit_width=4, num_outlier_channels=0,
                   outlier_bits=0, max_new_tokens=100):
    input_text = format_prompt(prompt)
    enc = tokenizer(input_text, return_tensors="pt")
    input_ids = enc.input_ids.to(DEVICE)
    attention_mask = enc.attention_mask.to(DEVICE)

    head_dim = model.config.hidden_size // model.config.num_attention_heads
    num_layers = model.config.num_hidden_layers

    tq_cache = TurboQuantCache(
        head_dim=head_dim,
        bit_width=bit_width,
        num_layers=num_layers,
        num_outlier_channels=num_outlier_channels,
        outlier_bits=outlier_bits,
        device=torch.device("cpu"),  # keep cache on CPU to save MPS memory
    )

    gc.collect()
    if DEVICE.type == "mps":
        torch.mps.empty_cache()

    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            past_key_values=tq_cache,
            use_cache=True,
            return_dict_in_generate=True,
        )
    elapsed = time.time() - start

    generated = outputs.sequences[0][input_ids.shape[1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True)
    kv_mem = outputs.past_key_values.get_memory_bytes()
    return text, elapsed, generated.shape[0], kv_mem

def main():
    print("=" * 70)
    print("  DeepSeekMath-7B-Instruct + TurboQuant — Apple M3 MPS")
    print("=" * 70 + "\n")

    model, tokenizer = load_model()
    head_dim = model.config.hidden_size // model.config.num_attention_heads

    configs = [
        ("Baseline (float16)",   "baseline",  {}),
        ("TurboQuant 4-bit",     "tq4",       {"bit_width": 4}),
        ("TurboQuant 3.5-bit",   "tq35",      {"bit_width": 3, "num_outlier_channels": 32, "outlier_bits": 4}),
    ]

    for prompt_idx, prompt in enumerate(MATH_PROMPTS):
        print(f"\n{'─' * 70}")
        print(f"Problem {prompt_idx + 1}: {prompt[:80]}...")
        print(f"{'─' * 70}")

        for label, key, kwargs in configs:
            print(f"\n  [{label}]")
            try:
                if key == "baseline":
                    text, elapsed, n_tokens = run_baseline(model, tokenizer, prompt)
                    kv_info = ""
                else:
                    text, elapsed, n_tokens, kv_mem = run_turboquant(
                        model, tokenizer, prompt, **kwargs
                    )
                    kv_info = f" | KV cache: {format_bytes(kv_mem)}"

                tps = n_tokens / elapsed if elapsed > 0 else 0
                print(f"  Time: {elapsed:.1f}s | Tokens: {n_tokens} | {tps:.1f} tok/s{kv_info}")
                print(f"  Output: {text[:400].strip()}{'...' if len(text) > 400 else ''}")
            except Exception as e:
                import traceback
                print(f"  ERROR: {e}")
                traceback.print_exc()

            gc.collect()
            if DEVICE.type == "mps":
                torch.mps.empty_cache()

    print(f"\n{'=' * 70}")
    print("  Done.")
    print(f"{'=' * 70}\n")

if __name__ == "__main__":
    main()
