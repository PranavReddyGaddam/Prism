import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
import time

MODEL_PATH = "./deepseek-model"
QUESTION = "If a train travels 120km in 2 hours, what is its average speed?"
ATTN_LAYER = 0
ATTN_HEAD = 0
MAX_TOKENS = 50
DEVICE = "mps"


def load_model():
    print("Loading DeepSeek model on MPS (Apple Silicon GPU)...")
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.float32,
        device_map=DEVICE
    )
    model.eval()
    load_time = time.time() - start
    print(f"Model loaded in {load_time:.2f} seconds")
    print(f"MPS memory: {torch.mps.current_allocated_memory() / 1024**3:.2f} GB\n")
    return tokenizer, model


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
    print(f"Generated {MAX_TOKENS} tokens in {gen_time:.2f} seconds")
    return output, decoded, inputs


def visualize_attention(model, tokenizer, inputs):
    print("\nGenerating attention heatmap...")
    start = time.time()

    model_cpu = model.to("cpu")
    inputs_cpu = {k: v.to("cpu") for k, v in inputs.items()}
    model_cpu.config._attn_implementation = "eager"

    with torch.no_grad():
        attn_outputs = model_cpu(**inputs_cpu, output_attentions=True)

    attentions = attn_outputs.attentions
    layer_attn = attentions[ATTN_LAYER][0, ATTN_HEAD].cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(inputs_cpu["input_ids"][0])

    model.to(DEVICE)

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
    plt.title(f"DeepSeek-R1-1.5B (MPS) Attention - Layer {ATTN_LAYER}, Head {ATTN_HEAD}")
    plt.xlabel("Key Tokens")
    plt.ylabel("Query Tokens")
    plt.tight_layout()
    plt.savefig("deepseek_mps_attention.png", dpi=150)
    plt.close()
    print(f"Attention heatmap saved in {time.time() - start:.2f} seconds")


def analyze_token_confidence(tokenizer, generation_output):
    print("\nToken confidence analysis:")
    scores = generation_output.scores
    sequence = generation_output.sequences[0]
    generated_ids = sequence[-len(scores):]

    for idx, score in enumerate(scores[:15]):
        prob = torch.softmax(score[0], dim=-1)
        token_id = generated_ids[idx]
        confidence = prob[token_id].item()
        token_str = tokenizer.decode([token_id])
        bar = "█" * int(confidence * 20)
        print(f"Token: {token_str:<12} | Confidence: {confidence:>6.2%} {bar}")


def logit_lens_analysis(model, tokenizer, inputs):
    print("\nLogit lens analysis (first 5 layers):")
    # Move inputs to CPU for stable float32 computation
    inputs_cpu = {k: v.to("cpu") for k, v in inputs.items()}
    model_cpu = model.to("cpu")
    with torch.no_grad():
        outputs = model_cpu(**inputs_cpu, output_hidden_states=True)
    hidden_states = outputs.hidden_states
    lm_head = model_cpu.lm_head.weight.float()
    last_token_id = inputs_cpu["input_ids"][0, -1]
    model.to(DEVICE)

    lm_head_cpu = lm_head.cpu()
    for layer_idx in range(min(5, len(hidden_states))):
        state = hidden_states[layer_idx][0, -1].float().cpu()
        logits = torch.matmul(state, lm_head_cpu.T)
        probs = torch.softmax(logits, dim=-1)
        top_prob, top_id = torch.topk(probs, 1)
        top_token = tokenizer.decode([top_id.item()])
        true_token = tokenizer.decode([last_token_id])
        print(f"Layer {layer_idx:02d}: predicts '{top_token}' (prob={top_prob.item():.2%}) vs true token '{true_token}'")


def gradient_token_attribution(model, tokenizer, inputs):
    print("\nComputing gradient-based attribution...")
    start = time.time()

    input_ids = inputs["input_ids"][:, -10:]
    attention_mask = inputs.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask[:, -10:]

    embeddings = model.get_input_embeddings()(input_ids).to(torch.float32).detach()
    embeddings.requires_grad_()

    outputs = model(
        inputs_embeds=embeddings.to(next(model.parameters()).dtype),
        attention_mask=attention_mask
    )
    logits = outputs.logits[0, -1, :].float()
    target_logit = logits.max()

    model.zero_grad()
    target_logit.backward()

    gradients = embeddings.grad[0]
    token_scores = (gradients * embeddings[0]).sum(dim=-1)

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    print("Attribution scores (last 10 tokens):")
    for token, score in zip(tokens, token_scores):
        print(f"  {token:<15} -> {score.item(): .4f}")
    print(f"Computed in {time.time() - start:.2f} seconds")


def hidden_state_summary(model, tokenizer, inputs):
    print("\nHidden state norms per layer:")
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states
    for idx, state in enumerate(hidden_states):
        norm = state.float().norm(dim=-1).mean().item()
        print(f"  Layer {idx:02d}: norm = {norm:.4f}")


def main():
    tokenizer, model = load_model()

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Layers: {model.config.num_hidden_layers}, Heads: {model.config.num_attention_heads}, Hidden: {model.config.hidden_size}\n")

    gen_output, decoded_text, inputs = run_generation(tokenizer, model, QUESTION)

    print("\n=== Model Response ===")
    print(decoded_text)

    if "<think>" in decoded_text and "</think>" in decoded_text:
        thinking = decoded_text.split("<think>")[1].split("</think>")[0].strip()
        final = decoded_text.split("</think>")[-1].strip()
        print("\n--- Chain of Thought ---")
        print(thinking)
        print("\n--- Final Answer ---")
        print(final)

    analyze_token_confidence(tokenizer, gen_output)
    logit_lens_analysis(model, tokenizer, inputs)
    gradient_token_attribution(model, tokenizer, inputs)
    hidden_state_summary(model, tokenizer, inputs)

    print(f"\nFinal MPS memory: {torch.mps.current_allocated_memory() / 1024**3:.2f} GB")
    print("DeepSeek MPS explainability analysis complete!")


if __name__ == "__main__":
    main()
