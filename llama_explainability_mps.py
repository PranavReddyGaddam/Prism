import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple
import time

MODEL_PATH = "./llama-3.2-3b-instruct"
QUESTION = "If a train travels 120km in 2 hours, what is its average speed?"
ATTN_LAYER = 0
ATTN_HEAD = 0
MAX_TOKENS = 30  # Reduced for speed
DEVICE = "mps"


def load_model():
    print("Loading tokenizer and model on MPS...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.float16,
        device_map=DEVICE
    )
    model.eval()
    print(f"Model loaded on {DEVICE}.")
    print(f"MPS memory: {torch.mps.current_allocated_memory() / 1024**3:.2f} GB\n")
    return tokenizer, model


def build_prompt(tokenizer, question: str) -> str:
    messages = [{"role": "user", "content": question}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


def run_generation(tokenizer, model, prompt: str):
    print("Generating response...")
    start = time.time()
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_TOKENS,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    gen_time = time.time() - start
    decoded = tokenizer.decode(output.sequences[0], skip_special_tokens=True)
    print(f"Generated in {gen_time:.2f} seconds")
    return output, decoded, inputs


def visualize_attention(model, tokenizer, inputs):
    print("Generating attention heatmap...")
    start = time.time()
    
    # Use CPU for attention extraction (MPS doesn't support output_attentions well)
    model_cpu = model.to("cpu")
    inputs_cpu = {k: v.to("cpu") for k, v in inputs.items()}
    model_cpu.config._attn_implementation = "eager"
    
    with torch.no_grad():
        attn_outputs = model_cpu(**inputs_cpu, output_attentions=True)
    
    attentions = attn_outputs.attentions
    layer_attn = attentions[ATTN_LAYER][0, ATTN_HEAD].cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(inputs_cpu["input_ids"][0])

    # Move model back to MPS
    model.to(DEVICE)

    # Show last 12 tokens for clarity
    N = min(12, len(tokens))
    layer_attn = layer_attn[-N:, -N:]
    tokens = tokens[-N:]

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        layer_attn,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap="viridis",
        annot=True,
        fmt=".2f"
    )
    plt.title(f"Llama-3.2-3B-Instruct (MPS) Attention - Layer {ATTN_LAYER}, Head {ATTN_HEAD}")
    plt.xlabel("Key Tokens")
    plt.ylabel("Query Tokens")
    plt.tight_layout()
    plt.savefig("llama_mps_attention.png", dpi=150)
    plt.close()
    print(f"Attention heatmap saved in {time.time() - start:.2f} seconds\n")


def analyze_token_confidence(tokenizer, generation_output):
    print("\nToken confidence analysis:")
    scores = generation_output.scores
    sequence = generation_output.sequences[0]
    generated_ids = sequence[-len(scores):]

    for idx, score in enumerate(scores[:15]):  # Show first 15 tokens
        prob = torch.softmax(score[0], dim=-1)
        token_id = generated_ids[idx]
        confidence = prob[token_id].item()
        token_str = tokenizer.decode([token_id])
        bar = "█" * int(confidence * 20)
        print(f"Token: {token_str:<12} | Confidence: {confidence:>6.2%} {bar}")


def logit_lens_analysis(model, tokenizer, inputs):
    print("\nLogit lens analysis (first 3 layers):")
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states
    lm_head = model.lm_head.weight
    last_token_id = inputs["input_ids"][0, -1]

    for layer_idx in range(min(3, len(hidden_states))):
        state = hidden_states[layer_idx][0, -1].float()
        logits = torch.matmul(state, lm_head.float().T)
        probs = torch.softmax(logits, dim=-1)
        top_prob, top_id = torch.topk(probs, 1)
        top_token = tokenizer.decode([top_id.item()])
        true_token = tokenizer.decode([last_token_id])
        print(f"Layer {layer_idx:02d}: predicts '{top_token}' (prob={top_prob.item():.2%}) vs true token '{true_token}'")


def gradient_token_attribution_fast(model, tokenizer, inputs):
    print("\nComputing gradient-based attribution (fast approximation)...")
    start = time.time()
    
    # Use only last 10 tokens for speed
    input_ids = inputs["input_ids"][:, -10:]
    attention_mask = inputs.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask[:, -10:]

    embeddings = model.get_input_embeddings()(input_ids).to(torch.float32).detach()
    embeddings.requires_grad_()

    outputs = model(inputs_embeds=embeddings.to(next(model.parameters()).dtype), attention_mask=attention_mask)
    logits = outputs.logits[0, -1, :].float()
    target_logit = logits.max()  # Use max logit instead of specific token

    model.zero_grad()
    target_logit.backward()

    gradients = embeddings.grad[0]
    token_scores = (gradients * embeddings[0]).sum(dim=-1)

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    print("Fast attribution scores (last 10 tokens):")
    for token, score in zip(tokens, token_scores):
        print(f"  {token:<15} -> {score.item(): .4f}")
    
    print(f"Computed in {time.time() - start:.2f} seconds")


def main():
    tokenizer, model = load_model()
    
    print(f"Model specs: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    prompt = build_prompt(tokenizer, QUESTION)
    gen_output, decoded_text, inputs = run_generation(tokenizer, model, prompt)

    print("\n=== Model Response ===")
    print(decoded_text)

    visualize_attention(model, tokenizer, inputs)
    analyze_token_confidence(tokenizer, gen_output)
    logit_lens_analysis(model, tokenizer, inputs)
    gradient_token_attribution_fast(model, tokenizer, inputs)

    print(f"\nFinal MPS memory: {torch.mps.current_allocated_memory() / 1024**3:.2f} GB")
    print("\nLlama MPS explainability analysis complete!")


if __name__ == "__main__":
    main()
