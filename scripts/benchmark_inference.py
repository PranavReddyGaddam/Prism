#!/usr/bin/env python3
"""
Load a local HF causal LM and time a single generate() after warmup.
Usage on RunPod (venv with torch + transformers + accelerate):

  Stack hints (RunPod / torch 2.4.x):
  - Avoid transformers 5.8+ on torch 2.4.1 (torch.library.custom_op errors).
  - Use transformers>=4.46,<5 for Qwen / DeepSeek / InternLM; **Phi-4-mini-reasoning
    needs transformers>=4.51.3** (see HF model card). Try: pip install "transformers==4.51.3"

  python scripts/benchmark_inference.py /workspace/models/qwen25-math-7b
  python scripts/benchmark_inference.py /workspace/models/phi4-mini --max-new-tokens 128
  python scripts/benchmark_inference.py /workspace/models/internlm2-math-plus-7b --trust-remote-code
"""

from __future__ import annotations

import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    p = argparse.ArgumentParser(description="HF local inference + timing")
    p.add_argument(
        "model_dir",
        help="Local snapshot dir (e.g. /workspace/models/qwen25-math-7b)",
    )
    p.add_argument(
        "--prompt",
        default="Solve step by step: What is 17 × 23?",
        help="Plain prompt string (add chat templates later if needed).",
    )
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
        help="Number of short generations before the timed run",
    )
    p.add_argument("--trust-remote-code", action="store_true")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available; run on the GPU pod.")

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(
        f"Loading {args.model_dir!r} (dtype={dtype}, trust_remote_code={args.trust_remote_code})..."
    )
    t_load = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(
        args.model_dir,
        trust_remote_code=args.trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()
    torch.cuda.synchronize()
    print(f"Load wall time: {time.perf_counter() - t_load:.2f}s")

    inputs = tok(args.prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    in_len = inputs["input_ids"].shape[1]
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    def generate_n(n_new: int) -> torch.Tensor:
        with torch.inference_mode():
            return model.generate(
                **inputs,
                max_new_tokens=n_new,
                do_sample=False,
                pad_token_id=pad_id,
            )

    warm_n = min(16, args.max_new_tokens)
    for _ in range(max(0, args.warmup_runs)):
        out = generate_n(warm_n)
        torch.cuda.synchronize()
    del out

    print(
        f"Timed generate (max_new_tokens={args.max_new_tokens}, warmup_runs={args.warmup_runs})..."
    )
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    output_ids = generate_n(args.max_new_tokens)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    new_tokens = int(output_ids.shape[1] - in_len)
    tps = new_tokens / elapsed if elapsed > 0 else 0.0

    text = tok.decode(output_ids[0], skip_special_tokens=True)

    print("---")
    print(f"New tokens (actual): {new_tokens}")
    print(f"Wall time (generate only): {elapsed:.3f}s")
    print(f"Throughput: {tps:.2f} new tokens/s")
    print(f"Latency per new token: {1000.0 * elapsed / max(new_tokens, 1):.2f} ms/token")
    print("--- Output (first 1200 chars) ---")
    print(text[:1200])
    if len(text) > 1200:
        print("... [truncated]")


if __name__ == "__main__":
    main()
